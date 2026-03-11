import openai
from src.config import OPENAI_API_KEY
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize OpenAI client
if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here":
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None
    logger.warning("OPENAI_API_KEY is not set. Response generation will fail.")

class ResponseGenerator:
    """
    Converts retrieved vehicle rows into a natural language recommendation using LLM.
    """
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def generate(self, query: str, vehicle_rows: list[dict]) -> str:
        """
        Generate natural language recommendation from rows.
        """
        if not client:
            raise ValueError("OpenAI API key is not configured.")

        if not vehicle_rows:
            return "I couldn't find any tyre recommendations for that query."

        # Group vehicle variants by tyres
        from collections import defaultdict
        
        # Structure: (brand, model, rec_tyre, upsize_tyre) -> list of variants
        grouped_data = defaultdict(list)
        
        for row in vehicle_rows:
            brand = row.get('vehicle-brand', 'NA')
            model = row.get('vehicle-model', 'NA')
            variant = row.get('vehicle-variant', 'NA')
            
            rec_tyre = row.get('recommended-tyre', row.get('recommended-sku.1', 'None'))
            if str(rec_tyre).strip() in ["NA", "None", "", "#N/A"]:
                rec_tyre = row.get("recommended-sku.1", "None")
                
            upsize_tyre = row.get('upsize-tyre', row.get('upsize-sku.1', 'None'))
            if str(upsize_tyre).strip() in ["NA", "None", "", "#N/A"]:
                upsize_tyre = row.get("upsize-sku.1", "None")
                
            grouped_data[(brand, model, rec_tyre, upsize_tyre)].append(variant)

        # Format grouped data for the prompt
        rows_text = ""
        for i, ((brand, model, rec_tyre, upsize_tyre), variants) in enumerate(grouped_data.items(), 1):
            variants_str = ", ".join(variants)
            rows_text += (
                f"Group {i}:\n"
                f"- Brand: {brand}\n"
                f"- Model: {model}\n"
                f"- Variants: {variants_str}\n"
                f"- Recommended Tyre: {rec_tyre}\n"
                f"- Upsize Tyre: {upsize_tyre}\n\n"
            )

        prompt = f"""You are an automotive tyre recommendation expert.

User query:
{query}

Vehicle data (already grouped):
{rows_text}

Write the recommendation in a professional and natural tone similar to an automotive product guide.

Rules:

1. Avoid repeating the same sentence structure for every variant group.
2. Combine information smoothly into short paragraphs instead of mechanical sentence blocks.
3. Use varied phrasing while clearly stating the recommended tyre and optional upsize where applicable.
4. If no upsize tyre exists, do not mention upsize.
5. CRITICAL: Do NOT mention the load index or speed rating in the tyre names (like 94Y, 100V, XL, SL, CALM). Only mention the tyre size and the brand/pattern name (e.g. "225/45R17 SportDrive TL").
6. Keep the response concise. Do not add general safety advice.
7. Also do not add any information from your side.

Example style:

The Hyundai Verna uses different tyre specifications depending on the variant. Models such as 1.6 I ABS, I (Petrol), and 1.6 XI ABS are equipped with 185/65R14 Milaze X3 TL tyres, with an optional upsize to 185/55R16 SecuraDrive TL.
Variants including 1.4 VTVT and 1.6 VTVT S use 185/65R15 SecuraDrive TL, which can be upsized to 195/60R15 SecuraDrive TL.
Higher-spec variants such as 1.6 VTVT AT S Option and 1.6 VTVT S Option are fitted with 195/55R16 SecuraDrive TL tyres."""

        logger.info("Sending prompt to OpenAI...")
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful tyre recommendation expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=1
            )
            answer = response.choices[0].message.content.strip()
            logger.info("Received generated response.")
            return answer
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise e
