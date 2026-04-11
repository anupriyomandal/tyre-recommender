import re
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

# Known tyre brand name normalizations (ALL CAPS → proper case)
_TYRE_BRAND_FIXES = {
    "SECURADRIVE SUV": "SecuraDrive SUV",
    "SECURADRIVE": "SecuraDrive",
    "SPORTDRIVE": "SportDrive",
    "CROSSDRIVE": "CrossDrive",
    "ENERGYDRIVE EV": "EnergyDrive EV",
    "ENERGYDRIVE": "EnergyDrive",
    "MILAZE X5": "Milaze X5",
    "MILAZE X3": "Milaze X3",
    "MILAZE": "Milaze",
    "SD004": "SD004",
}

def _normalize_tyre_name(name: str) -> str:
    """Strip everything after 'TL', fix brand casing, and clean extra spaces."""
    name = name.strip()
    # Keep only up to and including "TL" — everything after is load/speed ratings
    name = re.sub(r'(?<=\bTL\b).*$', '', name, flags=re.IGNORECASE).strip()
    # Fix known brand name casing
    for upper, proper in _TYRE_BRAND_FIXES.items():
        name = re.sub(re.escape(upper), proper, name, flags=re.IGNORECASE)
    # Collapse multiple spaces
    name = re.sub(r'  +', ' ', name)
    return name.strip()


class ResponseGenerator:
    """
    Converts retrieved vehicle rows into a natural language recommendation using LLM.
    """
    def __init__(self, model: str = "gpt-5.4-mini"):
        self.model = model
        self.unknown_answer = "I don't exactly know"

    def generate(self, query: str, vehicle_rows: list[dict], history: list[dict] | None = None) -> str:
        """
        Generate natural language recommendation from rows.
        history: list of {"role": "user"|"assistant", "content": "..."} messages
        """
        if not client:
            raise ValueError("OpenAI API key is not configured.")

        # No vehicle rows — conversational mode (answer from history)
        if not vehicle_rows:
            return self._generate_conversational(query, history)

        from collections import defaultdict

        # Group vehicle variants by (brand, model, rec_tyre, upsize_tyre)
        grouped_data = defaultdict(list)

        for row in vehicle_rows:
            brand = str(row.get('vehicle-brand', 'NA')).strip().title()
            model = str(row.get('vehicle-model', 'NA')).strip().title()
            variant = str(row.get('vehicle-variant', 'NA')).strip()

            rec_tyre = row.get('recommended-tyre', row.get('recommended-sku.1', 'None'))
            if str(rec_tyre).strip() in ["NA", "None", "", "#N/A", "nan"]:
                rec_tyre = row.get("recommended-sku.1", "None")
            rec_tyre = _normalize_tyre_name(str(rec_tyre)) if str(rec_tyre).strip() not in ["None", "nan", ""] else "None"

            upsize_tyre = row.get('upsize-tyre', row.get('upsize-sku.1', 'None'))
            if str(upsize_tyre).strip() in ["NA", "None", "", "#N/A", "nan"]:
                upsize_tyre = row.get("upsize-sku.1", "None")
            upsize_tyre = _normalize_tyre_name(str(upsize_tyre)) if str(upsize_tyre).strip() not in ["None", "nan", ""] else "None"

            if variant not in grouped_data[(brand, model, rec_tyre, upsize_tyre)]:
                grouped_data[(brand, model, rec_tyre, upsize_tyre)].append(variant)

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

User question:
{query}

Vehicle tyre fitment data (grouped by tyre):
{rows_text}

Instructions:

Answer the user's specific question using only the vehicle data above and the conversation history.

Rules:
1. Start directly with the answer. No introductions or preamble.
2. End after the last point. No summaries, conclusions, or filler.
3. Group variants that share the same tyre. Use natural phrases like "For the base models...", "The top-spec variants...", etc. BANNED word: "Most". Every distinct tyre size in the data must be mentioned.
4. Use natural, varied sentence structures — avoid repeating the same pattern for each paragraph.
5. Only mention upsize options if the user asks about upsize or if there is an upsize available. Never say "no upsize available" — just omit it.
6. CRITICAL: Never include load index or speed ratings (like 94Y, 100V, XL). Only mention tyre size and pattern name (e.g. "225/45R17 SportDrive TL").
7. CRITICAL: Format all tyre names and vehicle names in bold using HTML <b>tags</b>. BANNED: Markdown **asterisks**. Never use **bold**.
8. Do not add safety advice, driving tips, or anything beyond the tyre recommendation.
9. If the user asks about upsize options: focus only on upsizes from the data. If no upsize tyre is listed for any group, say simply "There are no upsize options listed for this vehicle." Do NOT say "I don't exactly know" in this case.
10. If the data and history genuinely cannot answer the question at all, reply with exactly: I don't exactly know
11. CRITICAL — Conversation continuity: If the conversation history already established a specific variant or vehicle (e.g. the user said "Verna SX" and you already answered about it), STAY FOCUSED on that specific variant for follow-up questions. Do NOT broaden your answer to all variants. The user's follow-up ("is there an upsize?") refers to the vehicle they already asked about, not every variant in the data.

Platform benefit — if a tyre belongs to one of these CEAT platforms, weave in the benefit naturally (once only, not repeated):
- Milaze X5: high mileage, long tread life
- Milaze X3: mileage focused, reliable everyday performance
- SecuraDrive: strong wet grip, improved braking
- SecuraDrive SUV: braking performance and stability for SUVs
- SportDrive: cornering grip, steering precision, high-speed stability
- CrossDrive: off-road traction and durability
- EnergyDrive EV: optimised for electric vehicles

Example of good output:
Several <b>Hyundai Verna</b> variants — including the 1.6 I ABS, I (Petrol), and 1.6 XI ABS — come fitted with <b>185/65R14 Milaze X3 TL</b>, a mileage-focused tyre for reliable everyday use. An upsize to <b>185/55R16 SecuraDrive TL</b> is available for better wet grip and braking.

The 1.4 VTVT and 1.6 VTVT S use <b>185/65R15 SecuraDrive TL</b>, with an upsize option of <b>195/60R15 SecuraDrive TL</b>."""

        logger.info("Sending prompt to OpenAI...")
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a tyre recommendation expert. Use only the provided vehicle data and conversation history. "
                        "Answer the user's specific question directly — if they ask about upsize, focus on upsize; "
                        "if they ask for a recommendation, give one. "
                        "IMPORTANT: If the conversation has already narrowed to a specific variant (e.g. user asked about 'Verna SX'), "
                        "keep all follow-up answers focused on that variant only — do NOT broaden to all variants. "
                        "If the data is insufficient, respond exactly with: I don't exactly know. "
                        "Always use <b>HTML bold tags</b> for tyre names. Never use Markdown **asterisks**."
                    )
                }
            ]

            if history:
                for msg in history:
                    if msg.get("role") == "assistant":
                        cleaned = re.sub(r'<[^>]+>', '', msg["content"])
                        messages.append({"role": "assistant", "content": cleaned})
                    else:
                        messages.append(msg)

            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3
            )
            answer = response.choices[0].message.content.strip()
            logger.info("Received generated response.")

            # Post-process: convert any stray **markdown** to <b>HTML</b>
            answer = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', answer)

            return answer
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise e

    def _generate_conversational(self, query: str, history: list[dict] | None) -> str:
        """Handle follow-up or conversational queries that have no new vehicle data."""
        if not history:
            return "I can help with tyre recommendations. Please share your vehicle's make and model."

        logger.info("Generating conversational reply from history (no new vehicle rows).")
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a tyre recommendation expert. Answer the user's follow-up question "
                        "using only the conversation history above. Keep your answer concise and factual. "
                        "Do not invent data. If you cannot answer from the history, say so simply. "
                        "Use <b>HTML bold tags</b> for tyre names. Never use Markdown **asterisks**."
                    )
                }
            ]
            for msg in history:
                if msg.get("role") == "assistant":
                    cleaned = re.sub(r'<[^>]+>', '', msg["content"])
                    messages.append({"role": "assistant", "content": cleaned})
                else:
                    messages.append(msg)
            messages.append({"role": "user", "content": query})

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3
            )
            answer = response.choices[0].message.content.strip()
            answer = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', answer)
            return answer
        except Exception as e:
            logger.error(f"Conversational reply failed: {e}")
            raise e
