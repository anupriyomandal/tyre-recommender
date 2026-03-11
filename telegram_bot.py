"""
Telegram bot that forwards user messages to the Tyre Recommender API
and returns the natural-language recommendation.

Required env vars:
    TELEGRAM_BOT_TOKEN  – Bot token from @BotFather
    API_URL             – Full URL to the /ask endpoint
                          (e.g. https://tyre-api-production.up.railway.app/ask)
"""

import os
import logging
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_URL = os.getenv("API_URL")   # e.g. https://your-railway-url/ask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    await update.message.reply_text(
        "👋 Welcome to the Tyre Recommender Bot!\n\n"
        "Just type the name of a vehicle (e.g. 'tyre for Verna') "
        "and I'll recommend the right tyres for you."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Forward the user's message to the API and return the answer."""
    user_query = update.message.text
    logger.info(f"Received query: {user_query}")

    # Initialize chat history if not present
    if "history" not in context.chat_data:
        context.chat_data["history"] = []

    # Show typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )

    try:
        response = requests.post(
            API_URL,
            json={
                "query": user_query,
                "history": context.chat_data["history"]
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        answer = data.get("answer", "Sorry, I could not generate a recommendation.")

        # Store exchange in history
        context.chat_data["history"].append({"role": "user", "content": user_query})
        context.chat_data["history"].append({"role": "assistant", "content": answer})

        # Keep last 10 exchanges (20 messages)
        if len(context.chat_data["history"]) > 20:
            context.chat_data["history"] = context.chat_data["history"][-20:]

    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        answer = "⚠️ Sorry, the recommendation service is currently unavailable. Please try again later."

    await update.message.reply_text(answer, parse_mode="HTML")


def main():
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set in the environment.")
    if not API_URL:
        raise ValueError("API_URL is not set in the environment.")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Telegram bot is starting...")
    app.run_polling()


if __name__ == "__main__":
    main()
