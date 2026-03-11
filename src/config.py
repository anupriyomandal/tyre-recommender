import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Files
DEFAULT_CSV_PATH = DATA_DIR / "tyres.csv"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.pkl"

# Model settings
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
