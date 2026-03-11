import faiss
import pickle
import numpy as np
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FaissIndexer:
    """
    Manages the FAISS index for vector search.
    """
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = None
        self.metadata = []

    def create_index(self):
        """Create a FAISS IndexFlatIP (Inner Product)."""
        logger.info(f"Creating FAISS IndexFlatIP with dimension {self.dimension}")
        self.index = faiss.IndexFlatIP(self.dimension)

    def add_embeddings(self, embeddings: np.ndarray, metadata: list[dict]):
        """
        Add embeddings and corresponding metadata.
        For inner product to calculate cosine similarity, vectors should be normalized.
        """
        if self.index is None:
            self.create_index()
            
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata records")

        logger.info(f"Adding {len(embeddings)} vectors to index...")
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        logger.info(f"Index now contains {self.index.ntotal} vectors.")

    def save_index(self, index_path: Path | str, metadata_path: Path | str):
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            logger.warning("No index to save.")
            return

        logger.info(f"Saving FAISS index to {index_path}")
        faiss.write_index(self.index, str(index_path))
        
        logger.info(f"Saving metadata to {metadata_path}")
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def load_index(self, index_path: Path | str, metadata_path: Path | str):
        """Load FAISS index and metadata from disk."""
        index_p = Path(index_path)
        meta_p = Path(metadata_path)

        if not index_p.exists() or not meta_p.exists():
            raise FileNotFoundError("Index or metadata file not found.")

        logger.info(f"Loading FAISS index from {index_p}")
        self.index = faiss.read_index(str(index_p))
        
        logger.info(f"Loading metadata from {meta_p}")
        with open(meta_p, 'rb') as f:
            self.metadata = pickle.load(f)
            
        logger.info(f"Successfully loaded index with {self.index.ntotal} vectors.")
