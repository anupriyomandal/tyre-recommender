from pathlib import Path
import re
from src.search.vector_search import VectorSearch
from src.llm.response_generator import ResponseGenerator
from src.config import (
    BM25_MIN_SCORE,
    OVERLAP_THRESHOLD_SHORT_QUERY,
    OVERLAP_THRESHOLD_MEDIUM_QUERY,
    OVERLAP_THRESHOLD_LONG_QUERY,
    VECTOR_SIMILARITY_MIN,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "i",
    "in",
    "is",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "show",
    "tell",
    "the",
    "to",
    "tyre",
    "tyres",
    "what",
    "which",
    "with",
}

class TyreRecommender:
    """
    Main recommender pipeline orchestrating search and LLM generation.
    """
    def __init__(self, index_path: Path | str, metadata_path: Path | str):
        self.vector_search = VectorSearch(index_path, metadata_path)
        self.response_generator = ResponseGenerator()
        self.unknown_answer = "I don't exactly know"
        self.bm25_min_score = BM25_MIN_SCORE
        self.overlap_threshold_short = OVERLAP_THRESHOLD_SHORT_QUERY
        self.overlap_threshold_medium = OVERLAP_THRESHOLD_MEDIUM_QUERY
        self.overlap_threshold_long = OVERLAP_THRESHOLD_LONG_QUERY
        self.vector_similarity_min = VECTOR_SIMILARITY_MIN

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]

    def _required_overlap_threshold(self, query_tokens: list[str]) -> float:
        token_count = len(query_tokens)
        if token_count <= 2:
            return self.overlap_threshold_short
        if token_count <= 5:
            return self.overlap_threshold_medium
        return self.overlap_threshold_long

    def _query_is_brand_only_ambiguous(self, query: str, vehicle_rows: list[dict]) -> bool:
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return False

        # Focus on top-ranked rows to infer ambiguity.
        top_rows = vehicle_rows[:10]

        matched_brand_rows: list[dict] = []
        model_matched = False

        for row in top_rows:
            brand_tokens = set(self._tokenize(str(row.get("vehicle-brand", ""))))
            model_tokens = set(self._tokenize(str(row.get("vehicle-model", ""))))

            if query_tokens.intersection(brand_tokens):
                matched_brand_rows.append(row)

            if query_tokens.intersection(model_tokens):
                model_matched = True

        if not matched_brand_rows or model_matched:
            return False

        distinct_models = {
            str(row.get("vehicle-model", "")).strip().lower()
            for row in matched_brand_rows
            if str(row.get("vehicle-model", "")).strip()
        }

        # Brand mentioned but model absent while multiple models are possible.
        return len(distinct_models) > 1

    def _normalize_unknown_answer(self, answer: str) -> str:
        normalized = answer.strip().lower().rstrip(".!")
        if normalized in {
            "i don't exactly know",
            "sorry, i don't know that",
            "i dont exactly know",
        }:
            return self.unknown_answer
        return answer

    def _has_strong_context_match(self, query: str, vehicle_rows: list[dict]) -> bool:
        if not vehicle_rows:
            return False

        query_tokens = self._tokenize(query)
        required_overlap = self._required_overlap_threshold(query_tokens)

        top_window = vehicle_rows[:3]
        best_bm25 = max(float(row.get("bm25_score", 0.0)) for row in top_window)
        best_overlap = max(float(row.get("token_overlap", 0.0)) for row in top_window)
        best_similarity = max(float(row.get("similarity_score", 0.0)) for row in top_window)

        # Permissive guard:
        # - Pass if semantic match is strong (helps with spelling mistakes), OR
        # - pass lexical BM25 + overlap checks.
        lexical_pass = best_bm25 >= self.bm25_min_score and best_overlap >= required_overlap
        semantic_pass = best_similarity >= self.vector_similarity_min
        return semantic_pass or lexical_pass

    def recommend(self, query: str, history: list[dict] | None = None) -> str:
        """
        1 vector search
        2 retrieve vehicle rows
        3 send rows to ResponseGenerator with conversation history
        4 return natural language answer
        """
        # Build search query: for follow-ups, combine with previous user query
        search_query = query
        if history:
            # Get the last user message from history to provide context
            previous_user_msgs = [m["content"] for m in history if m["role"] == "user"]
            if previous_user_msgs:
                search_query = f"{previous_user_msgs[-1]} {query}"
        
        logger.info(f"Starting recommendation workflow for query: '{query}' (search: '{search_query}')")
        
        # 1 & 2. Vector search to retrieve vehicle rows
        try:
            vehicle_rows = self.vector_search.search(search_query, k=10)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"Error during search: {e}"

        if not self._has_strong_context_match(search_query, vehicle_rows):
            logger.info("No sufficiently relevant context found. Returning unknown answer.")
            return self.unknown_answer
        if self._query_is_brand_only_ambiguous(search_query, vehicle_rows):
            logger.info("Brand-only ambiguous query detected. Returning unknown answer.")
            return self.unknown_answer

        # 3 & 4. Generate and return natural language answer
        try:
            answer = self.response_generator.generate(query, vehicle_rows, history=history)
            if not answer or not answer.strip():
                return self.unknown_answer
            return self._normalize_unknown_answer(answer)
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"Error creating recommendation: {e}"
