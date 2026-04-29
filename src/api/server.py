import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Ensure root directory is on the Python path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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
        answer = recommender.recommend(request.query, history=request.history)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"/ask error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream")
def ask_stream(request: QueryRequest):
    """
    Accept a natural-language query and stream a tyre recommendation via SSE.
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized.")

    def event_generator():
        try:
            for chunk in recommender.recommend_stream(request.query, history=request.history):
                # Escape newlines for SSE
                text = chunk.replace("\n", "\\n")
                yield f"data: {text}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"/ask/stream error: {e}")
            yield f"data: Error: {str(e).replace(chr(10), ' ')}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
