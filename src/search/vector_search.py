from pathlib import Path
from src.embeddings.embedding_model import EmbeddingModel
from src.indexing.faiss_indexer import FaissIndexer
from src.utils.logger import get_logger

logger = get_logger(__name__)

class VectorSearch:
    """
    Handles semantic search over the tyre database.
    """
    def __init__(self, index_path: Path | str, metadata_path: Path | str):
        self.embedding_model = EmbeddingModel()
        # text-embedding-3-small uses 1536 dimensions
        self.indexer = FaissIndexer(dimension=1536) 
        self.indexer.load_index(index_path, metadata_path)

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        1 convert query to embedding
        2 search FAISS index
        3 retrieve top matching rows
        4 return structured vehicle records
        """
        logger.info(f"Searching for query: '{query}'")
        
        # 1. Convert query to embedding
        query_embedding = self.embedding_model.encode_query(query)
        import faiss # needed for normalization
        faiss.normalize_L2(query_embedding)

        # 2. Search FAISS index
        logger.info(f"Finding top {k} matches in index...")
        distances, indices = self.indexer.index.search(query_embedding, k)

        # 3 & 4. Retrieve and return top matching rows
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue # not enough results
            
            record = self.indexer.metadata[idx]
            # optional: add score to record
            # record['similarity_score'] = float(distances[0][i])
            results.append(record)
            
        logger.info(f"Found {len(results)} matching records.")
        return results
