from pathlib import Path
import re
from src.embeddings.embedding_model import EmbeddingModel
from src.indexing.faiss_indexer import FaissIndexer
from src.utils.logger import get_logger
from src.utils.stopwords import STOPWORDS

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

    def _tokenize(self, text: str) -> list[str]:
        # Normalise Unicode multiplication sign (×) to ASCII x so "2×4" → "2x4"
        text = text.replace("\u00d7", "x")
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]

    def _record_to_text(self, record: dict) -> str:
        keys = [
            "category",
            "vehicle-brand",
            "vehicle-model",
            "vehicle-variant",
            "manufacturing-year",
            "recommended-sku.1",
            "upsize-sku.1",
            "others-sku.1",
        ]
        return " ".join(str(record.get(key, "")) for key in keys)

    def _token_overlap(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not query_tokens:
            return 0.0
        query_set = set(query_tokens)
        doc_set = set(doc_tokens)
        return len(query_set.intersection(doc_set)) / len(query_set)

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        1 convert query to embedding
        2 retrieve candidate set from FAISS
        3 return structured vehicle records sorted by overlap then similarity
        """
        logger.info(f"Searching for query: '{query}'")
        query_tokens = self._tokenize(query)

        # 1. Convert query to embedding
        query_embedding = self.embedding_model.encode_query(query)
        import faiss  # needed for normalization
        faiss.normalize_L2(query_embedding)

        # 2. Search FAISS index for a larger candidate pool
        candidate_k = min(max(k * 10, k), self.indexer.index.ntotal)
        logger.info(f"Finding top {candidate_k} vector candidates in index...")
        distances, indices = self.indexer.index.search(query_embedding, candidate_k)

        candidates: list[dict] = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue  # not enough results

            record = dict(self.indexer.metadata[idx])
            record_text = self._record_to_text(record)
            doc_tokens = self._tokenize(record_text)
            record["similarity_score"] = float(distances[0][i])
            record["token_overlap"] = self._token_overlap(query_tokens, doc_tokens)
            candidates.append(record)

        if not candidates:
            logger.info("No matching records found.")
            return []

        # Sort by overlap first (exact brand/model keyword match),
        # then by semantic similarity as tiebreaker.
        candidates.sort(
            key=lambda item: (
                item.get("token_overlap", 0.0),
                item.get("similarity_score", 0.0),
            ),
            reverse=True,
        )

        # 3. Select top-k distinct tyre groups, then return ALL rows that belong
        #    to those groups so the LLM sees every variant — not just one per group.
        def _group_key(row: dict) -> tuple:
            return (
                str(row.get("vehicle-brand", "")).strip().lower(),
                str(row.get("vehicle-model", "")).strip().lower(),
                str(row.get("recommended-sku.1", row.get("recommended-tyre", ""))).strip().lower(),
            )

        # Walk the ranked list and collect the first k unique groups (in rank order)
        selected_groups: list[tuple] = []
        seen_groups: set[tuple] = set()
        for row in candidates:
            gk = _group_key(row)
            if gk not in seen_groups:
                seen_groups.add(gk)
                selected_groups.append(gk)
            if len(selected_groups) >= k:
                break

        selected_set = set(selected_groups)

        # Return every candidate row whose group is in the selected set
        results = [row for row in candidates if _group_key(row) in selected_set]
        logger.info(f"Found {len(results)} rows across {len(selected_groups)} tyre groups ({len(candidates)} raw candidates).")
        return results
