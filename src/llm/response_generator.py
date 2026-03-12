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
        self.unknown_answer = "Sorry, I don't know that"

    def generate(self, query: str, vehicle_rows: list[dict], history: list[dict] | None = None) -> str:
        """
        Generate natural language recommendation from rows.
        history: list of {"role": "user"|"assistant", "content": "..."} messages
        """
        if not client:
            raise ValueError("OpenAI API key is not configured.")

        if not vehicle_rows:
            return self.unknown_answer

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

Write a concise, natural-sounding tyre recommendation as if you are a knowledgeable tyre advisor speaking to a customer.

Rules:

1. Start directly with the recommendation. No introductions.
2. End after the last recommendation. No summaries or conclusions.
3. Group variants logically if they share the same tyre, using natural phrases like "For the base models..." or "Several variants including [A] and [B] use...". CRITICAL: You are BANNED from using the word "Most". Do not use it under any circumstances. Ensure EVERY distinct tyre option provided in the vehicle data is mentioned in your response. Do not omit any tyre sizes.
4. Use natural, varied sentence structures. Do not repeat the same pattern for each paragraph.
5. If no upsize tyre exists, do not mention upsize.
6. CRITICAL: Do NOT include load index or speed ratings (like 94Y, 100V, XL, SL). Only mention tyre size and pattern name (e.g. "225/45R17 SportDrive TL").
7. CRITICAL: You MUST format all tyre names in bold using HTML <b>bold</b> tags (e.g., <b>215/60R17 SecuraDrive SUV TL</b>). You are BANNED from using Markdown asterisks (**). Do not use **bold** under any circumstances. Bold the vehicle brand and model on first mention.
8. Do not add safety advice, driving tips, or any information beyond the tyre recommendation and platform benefit below.
9. Use only the provided vehicle data. If the query cannot be answered strictly from this data, reply with exactly: Sorry, I don't know that

Platform benefit rule:

If a recommended tyre belongs to one of the CEAT platforms below, weave the platform benefit naturally into the recommendation in a few words. Do not repeat the same platform benefit more than once.

Platforms:
- Milaze X5: high mileage, long tread life
- Milaze X3: mileage focused, reliable everyday performance
- SecuraDrive: strong wet grip, improved braking
- SecuraDrive SUV: braking performance and stability for SUVs
- SportDrive: cornering grip, steering precision, high speed stability
- CALM: reduced road noise, quieter cabin
- CrossDrive: off-road traction and durability

Example:

Several <b>Hyundai Verna</b> variants — including the 1.6 I ABS, I (Petrol), and 1.6 XI ABS — come fitted with <b>185/65R14 Milaze X3 TL</b>, a mileage-focused tyre built for reliable everyday use. An upsize to <b>185/55R16 SecuraDrive TL</b> is available for those looking for better wet grip and braking.

The 1.4 VTVT and 1.6 VTVT S use <b>185/65R15 SecuraDrive TL</b>, with an upsize option to <b>195/60R15 SecuraDrive TL</b>.

Higher-spec variants like the 1.6 VTVT AT S Option run <b>195/55R16 SecuraDrive TL</b>."""

        logger.info("Sending prompt to OpenAI...")
        try:
            messages = [
                {"role": "system", "content": "You are a tyre recommendation expert. Use only provided context rows. If context is insufficient for the query, respond exactly with: Sorry, I don't know that. Always start your response directly with the first tyre recommendation. Never begin with generic introductions. Never end with generic summaries or conclusions. You MUST use <b>HTML tags</b> for tyre names."}
            ]
            
            # Append conversation history if available
            if history:
                messages.extend(history)
            
            # Append the current query with vehicle context
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3
            )
            answer = response.choices[0].message.content.strip()
            logger.info("Received generated response.")
            
            # Post-processing to defensively convert any stray markdown ** to HTML <b>
            import re
            # Replaces **text** with <b>text</b>
            answer = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', answer)
            
            return answer
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise e
