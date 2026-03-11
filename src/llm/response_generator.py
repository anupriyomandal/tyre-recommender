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

    def generate(self, query: str, vehicle_rows: list[dict], history: list[dict] | None = None) -> str:
        """
        Generate natural language recommendation from rows.
        history: list of {"role": "user"|"assistant", "content": "..."} messages
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
5. CRITICAL: Do NOT mention the load index or speed rating in the tyre names (like 94Y, 100V, XL, SL). Only mention the tyre size and the brand/pattern name (e.g. "225/45R17 SportDrive TL").
6. Keep the response concise. Do not add general safety advice.
7. Do not add any information beyond the tyre recommendation and the platform benefit explained below.
8. Format tyre names in bold using HTML <b>bold</b> tags. Also bold the vehicle brand and model name on first mention.

Platform benefit rule:

When a recommended tyre belongs to one of the CEAT platforms listed below, briefly explain the key benefit of that platform in one short sentence.  
If multiple tyres belong to the same platform, explain the platform benefit only once.

Platform knowledge:

Milaze X5  
High mileage platform designed for very long tread life and lower cost per kilometre.

Milaze X3  
Mileage focused tyre offering long tyre life and reliable everyday performance.

SecuraDrive  
Safety focused platform designed for strong wet grip, improved braking, and stable highway handling.

SecuraDrive SUV  
SUV specific platform designed to provide strong braking performance and stability for heavier vehicles.

SportDrive  
Performance platform designed for excellent cornering grip, precise steering response, and high speed stability.

CALM tyres  
Noise reduction platform designed to reduce road noise and provide a quieter and more comfortable cabin experience.

CrossDrive  
All terrain SUV platform designed to provide strong traction on rough roads and durability for challenging terrain.

Platform explanation rules:

- Mention the platform benefit only if that platform appears in the recommended tyre name.
- Keep the explanation short and natural, like a product catalogue description.
- Do not repeat the same platform explanation multiple times.

Example style:

The <b>Hyundai Verna</b> uses different tyre specifications depending on the variant. Models such as 1.6 I ABS, I (Petrol), and 1.6 XI ABS are equipped with <b>185/65R14 Milaze X3 TL</b> tyres, with an optional upsize to <b>185/55R16 SecuraDrive TL</b>.

Variants including 1.4 VTVT and 1.6 VTVT S use <b>185/65R15 SecuraDrive TL</b>, which can be upsized to <b>195/60R15 SecuraDrive TL</b>. The SecuraDrive platform focuses on strong braking performance and wet grip, helping deliver stable and confident highway driving.

Higher-spec variants such as 1.6 VTVT AT S Option and 1.6 VTVT S Option are fitted with <b>195/55R16 SecuraDrive TL</b> tyres."""

        logger.info("Sending prompt to OpenAI...")
        try:
            messages = [
                {"role": "system", "content": "You are a helpful tyre recommendation expert."}
            ]
            
            # Append conversation history if available
            if history:
                messages.extend(history)
            
            # Append the current query with vehicle context
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=1
            )
            answer = response.choices[0].message.content.strip()
            logger.info("Received generated response.")
            return answer
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise e
