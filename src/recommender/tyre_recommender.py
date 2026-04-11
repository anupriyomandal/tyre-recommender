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

GREETINGS = {"hello", "hi", "hey", "howdy", "greetings", "hiya"}
THANKS = {"thanks", "thank", "ok", "okay", "great", "cool", "perfect", "noted", "bye", "goodbye"}

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

    def _is_greeting(self, query: str) -> bool:
        tokens = set(re.findall(r"[a-z]+", query.lower()))
        return bool(tokens.intersection(GREETINGS)) and len(tokens) <= 4

    def _is_thanks_or_closing(self, query: str) -> bool:
        tokens = set(re.findall(r"[a-z]+", query.lower()))
        return bool(tokens.intersection(THANKS)) and len(tokens) <= 4

    def _query_is_brand_only_ambiguous(self, query: str, vehicle_rows: list[dict]) -> bool:
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return False

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

        return len(distinct_models) > 1

    def _get_brand_clarification(self, query: str, vehicle_rows: list[dict]) -> str:
        """Return a clarification message listing available models for the detected brand."""
        query_tokens = set(self._tokenize(query))
        brand_name = None
        models: list[str] = []

        for row in vehicle_rows[:20]:
            brand = str(row.get("vehicle-brand", "")).strip()
            brand_tokens = set(self._tokenize(brand))
            if query_tokens.intersection(brand_tokens):
                if brand_name is None:
                    brand_name = brand.title()
                model = str(row.get("vehicle-model", "")).strip()
                if model and model not in ("NA", "None", "") and model.upper() not in [m.upper() for m in models]:
                    models.append(model.title())

        models = models[:6]
        if brand_name and models:
            model_list = ", ".join(models)
            return f"Which {brand_name} model are you looking for? For example: {model_list}."
        elif brand_name:
            return f"Which {brand_name} model are you looking for? Please provide the model name."
        return "Could you tell me the make and model of your vehicle?"

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
        # Handle greetings
        if self._is_greeting(query):
            return "Hi! I'm your tyre recommendation assistant. Tell me your vehicle's make and model and I'll suggest the right tyres."

        # Handle closings/thanks
        if self._is_thanks_or_closing(query) and not history:
            return "You're welcome! Feel free to ask if you need tyre recommendations for any other vehicle."

        # Build search query: for follow-ups, combine with previous user query
        search_query = query
        if history:
            previous_user_msgs = [m["content"] for m in history if m["role"] == "user"]
            if previous_user_msgs:
                search_query = f"{previous_user_msgs[-1]} {query}"

        logger.info(f"Starting recommendation workflow for query: '{query}' (search: '{search_query}')")

        # 1 & 2. Vector search to retrieve vehicle rows
        try:
            vehicle_rows = self.vector_search.search(search_query, k=20)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"Error during search: {e}"

        # Check brand-only ambiguity FIRST — before the context match gate —
        # so "what's the tyre for Audi" always gets a model clarification
        # regardless of whether the similarity score crosses the threshold.
        if vehicle_rows and self._query_is_brand_only_ambiguous(search_query, vehicle_rows):
            logger.info("Brand-only ambiguous query detected. Returning model clarification.")
            return self._get_brand_clarification(search_query, vehicle_rows)

        has_match = self._has_strong_context_match(search_query, vehicle_rows)

        if not has_match:
            # If there's conversation history, let the LLM answer conversationally (e.g. follow-ups, thanks, comparisons)
            if history:
                logger.info("No strong context match but history present — routing to LLM for conversational reply.")
                try:
                    answer = self.response_generator.generate(query, [], history=history)
                    if answer and answer.strip():
                        return self._normalize_unknown_answer(answer)
                except Exception as e:
                    logger.error(f"Conversational fallback failed: {e}")
            # No vehicle data and no history — give a helpful redirect
            logger.info("No sufficiently relevant context found. Returning helpful redirect.")
            return "I can help you find the right tyres. Please share the make and model of your vehicle (e.g. Honda City, Maruti Swift, Toyota Fortuner)."

        # Filter to only rows that match the top-ranked brand+model to avoid
        # passing unrelated vehicles to the LLM when k=20 brings in noise.
        top_brand = str(vehicle_rows[0].get("vehicle-brand", "")).strip().lower()
        top_model = str(vehicle_rows[0].get("vehicle-model", "")).strip().lower()
        filtered_rows = [
            r for r in vehicle_rows
            if str(r.get("vehicle-brand", "")).strip().lower() == top_brand
            and str(r.get("vehicle-model", "")).strip().lower() == top_model
        ]
        if not filtered_rows:
            filtered_rows = vehicle_rows

        # 3 & 4. Generate and return natural language answer
        try:
            answer = self.response_generator.generate(query, filtered_rows, history=history)
            if not answer or not answer.strip():
                return self.unknown_answer
            return self._normalize_unknown_answer(answer)
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"Error creating recommendation: {e}"
