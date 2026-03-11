#!/bin/bash

# Start the FastAPI server in the background
uvicorn src.api.server:app --host 0.0.0.0 --port ${PORT:-8000} &

# Wait for the API to be ready
echo "Waiting for API server to start..."
sleep 5

# Set API_URL to localhost if not already set
export API_URL="${API_URL:-http://localhost:${PORT:-8000}/ask}"

# Start the Telegram bot in the foreground
echo "Starting Telegram bot..."
python telegram_bot.py
