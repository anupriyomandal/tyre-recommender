import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Ensure root directory is on the Python path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException
from src.api.schemas import QueryRequest
from src.config import FAISS_INDEX_PATH, METADATA_PATH
from src.recommender.tyre_recommender import TyreRecommender
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global recommender instance
recommender: TyreRecommender | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the FAISS index and TyreRecommender once at startup."""
    global recommender
    logger.info("Loading FAISS index and initializing TyreRecommender...")
    try:
        recommender = TyreRecommender(FAISS_INDEX_PATH, METADATA_PATH)
        logger.info("TyreRecommender is ready.")
    except Exception as e:
        logger.error(f"Failed to initialize TyreRecommender: {e}")
        raise RuntimeError(f"Startup failed: {e}")
    yield
    logger.info("Shutting down TyreRecommender API.")


app = FastAPI(
    title="Tyre Recommender API",
    description="Recommend tyres for vehicles using semantic search and LLM.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    """Simple health-check endpoint."""
    return {"status": "ok"}


@app.post("/ask")
def ask(request: QueryRequest):
    """
    Accept a natural-language query and return a tyre recommendation.
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized.")

    try:
        answer = recommender.recommend(request.query)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"/ask error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
