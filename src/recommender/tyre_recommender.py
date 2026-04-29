from pathlib import Path
import re
from src.search.vector_search import VectorSearch
from src.llm.response_generator import ResponseGenerator
from src.config import (
    OVERLAP_THRESHOLD_SHORT_QUERY,
    OVERLAP_THRESHOLD_MEDIUM_QUERY,
    OVERLAP_THRESHOLD_LONG_QUERY,
    VECTOR_SIMILARITY_MIN,
)
from src.utils.logger import get_logger
from src.utils.stopwords import STOPWORDS

logger = get_logger(__name__)

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
        self.overlap_threshold_short = OVERLAP_THRESHOLD_SHORT_QUERY
        self.overlap_threshold_medium = OVERLAP_THRESHOLD_MEDIUM_QUERY
        self.overlap_threshold_long = OVERLAP_THRESHOLD_LONG_QUERY
        self.vector_similarity_min = VECTOR_SIMILARITY_MIN

    def _tokenize(self, text: str) -> list[str]:
        # Normalise Unicode multiplication sign (×) to ASCII x so "2×4" → "2x4"
        text = text.replace("\u00d7", "x")
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

    # Keywords that signal the user wants a list of variants, not a tyre recommendation.
    # NOTE: "all" alone is too broad — "tyre for all variants" is a tyre query, not a
    # listing query.  Keep this set narrow to avoid false positives.
    _VARIANT_LIST_KEYWORDS = {
        "variants", "versions", "options", "trims", "trim",
        "available", "types", "different",
    }

    # If ANY of these words appear alongside a listing keyword, the user is asking
    # for tyre info, not a variant list — skip the listing branch entirely.
    _TYRE_INTENT_KEYWORDS = {
        "tyre", "tyres", "tire", "tires", "recommended", "recommend",
        "suggestion", "suggest", "fitted", "fit", "suitable",
    }

    def _collect_model_rows(
        self,
        query: str,
        vehicle_rows: list[dict],
        use_top_fallback: bool = False,
    ) -> list[dict]:
        """Return vehicle rows that match the model named in `query`.

        When `use_top_fallback` is True and the current query contains no model
        name (e.g. "tell me for all the versions"), fall back to the brand+model
        of the top-ranked row — which was retrieved using the history-augmented
        search query and therefore already reflects the correct vehicle.
        """
        query_tokens = {t for t in self._tokenize(query) if not self._is_year_token(t)}

        matched: list[dict] = []
        for row in vehicle_rows[:20]:
            model_tokens = set(self._tokenize(str(row.get("vehicle-model", ""))))
            if query_tokens.intersection(model_tokens):
                matched.append(row)

        if matched:
            return matched

        # Fallback: no model name in current query — use the top vehicle row's brand+model
        if use_top_fallback and vehicle_rows:
            top_brand = str(vehicle_rows[0].get("vehicle-brand", "")).strip().lower()
            top_model = str(vehicle_rows[0].get("vehicle-model", "")).strip().lower()
            return [
                r for r in vehicle_rows[:20]
                if str(r.get("vehicle-brand", "")).strip().lower() == top_brand
                and str(r.get("vehicle-model", "")).strip().lower() == top_model
            ]

        return []

    def _is_variant_listing_query(self, query: str, vehicle_rows: list[dict], has_history: bool = False) -> bool:
        """True if the user is asking 'what variants / all versions' style question.

        Returns False when the query also contains tyre-intent words (e.g.
        "what is the recommended tyre for all of these variants") — that is a
        tyre recommendation request, not a variant listing request.
        """
        raw_tokens = set(re.findall(r"[a-z]+", query.lower()))
        # Must contain at least one variant-listing keyword
        if not raw_tokens.intersection(self._VARIANT_LIST_KEYWORDS):
            return False
        # If the user is also asking about tyres, this is a tyre query, not a listing
        if raw_tokens.intersection(self._TYRE_INTENT_KEYWORDS):
            return False
        # Must be able to resolve a vehicle — either by name in query or via history context
        return bool(self._collect_model_rows(query, vehicle_rows, use_top_fallback=has_history))

    def _get_variant_list_response(self, query: str, vehicle_rows: list[dict]) -> str:
        """Return a message listing all known variants for the matched model."""
        model_rows = self._collect_model_rows(query, vehicle_rows, use_top_fallback=True)

        brand_name = None
        model_name = None
        variants: list[str] = []

        for row in model_rows:
            brand = str(row.get("vehicle-brand", "")).strip()
            model = str(row.get("vehicle-model", "")).strip()
            variant = str(row.get("vehicle-variant", "")).strip()

            if brand_name is None:
                brand_name = brand.title()
            if model_name is None:
                model_name = model.title()

            if variant and variant not in ("NA", "None", "", "nan") and variant.upper() not in [v.upper() for v in variants]:
                variants.append(variant.title())

        if brand_name and model_name and variants:
            variant_lines = "".join(f"<br>• {v}" for v in variants)
            return (
                f"The <b>{brand_name} {model_name}</b> is available in the following variants:{variant_lines}<br><br>"
                f"Which variant do you have? I'll suggest the right tyres for it."
            )
        elif brand_name and model_name:
            return f"I have tyre data for the {brand_name} {model_name}, but variant information isn't detailed enough. Could you tell me your specific variant?"
        return "Could you tell me the make and model of your vehicle so I can list its variants?"

    @staticmethod
    def _is_year_token(token: str) -> bool:
        """True if the token looks like a 4-digit manufacturing year (1900-2099)."""
        return bool(re.match(r"^(19|20)\d{2}$", token))

    # Words that signal the user explicitly wants ALL variants — skip the ambiguity
    # clarification and let the LLM answer with the full tyre breakdown.
    _ALL_VARIANTS_KEYWORDS = {"all", "every", "each", "these"}

    def _query_is_variant_ambiguous(self, query: str, vehicle_rows: list[dict], has_history: bool = False) -> bool:
        """True if model is matched, no variant is specified in query, and ≥3 distinct tyre groups exist.

        Returns False when the user explicitly asks for ALL variants (e.g.
        "tell me for all variants", "what is the tyre for all of these") —
        in that case the LLM should answer with the full tyre breakdown.

        Manufacturing years (e.g. '2005', '2019') are treated as neutral — they do NOT
        count as a variant match.  When `has_history` is True and the query has no model
        name, the top vehicle row is used as the subject (context carry-over).
        """
        # If user explicitly asks for ALL variants, don't ask for clarification —
        # let the LLM provide the full tyre breakdown.
        raw_tokens = set(re.findall(r"[a-z]+", query.lower()))
        if raw_tokens.intersection(self._ALL_VARIANTS_KEYWORDS) and raw_tokens.intersection(
            self._VARIANT_LIST_KEYWORDS | {"variant", "version"}
        ):
            return False

        non_year_tokens = {t for t in self._tokenize(query) if not self._is_year_token(t)}

        matched_model_rows = self._collect_model_rows(query, vehicle_rows, use_top_fallback=has_history)
        if not matched_model_rows:
            return False

        # Check whether any non-year query token matches a variant token
        variant_matched = False
        for row in matched_model_rows:
            variant_tokens = set(self._tokenize(str(row.get("vehicle-variant", ""))))
            if non_year_tokens and non_year_tokens.intersection(variant_tokens):
                variant_matched = True
                break

        if variant_matched:
            return False

        # Count distinct recommended tyre groups
        distinct_tyres = {
            str(row.get("recommended-sku.1", row.get("recommended-tyre", ""))).strip().lower()
            for row in matched_model_rows
            if str(row.get("recommended-sku.1", row.get("recommended-tyre", ""))).strip()
            not in ("", "nan", "#n/a", "na")
        }
        return len(distinct_tyres) >= 3

    def _get_variant_clarification(self, query: str, vehicle_rows: list[dict]) -> str:
        """Return a clarification message listing available variants for the matched model."""
        model_rows = self._collect_model_rows(query, vehicle_rows, use_top_fallback=True)

        brand_name = None
        model_name = None
        variants: list[str] = []

        for row in model_rows:
            brand = str(row.get("vehicle-brand", "")).strip()
            model = str(row.get("vehicle-model", "")).strip()
            variant = str(row.get("vehicle-variant", "")).strip()

            if brand_name is None:
                brand_name = brand.title()
            if model_name is None:
                model_name = model.title()

            if variant and variant not in ("NA", "None", "", "nan") and variant.upper() not in [v.upper() for v in variants]:
                variants.append(variant.title())

        variants = variants[:6]
        if brand_name and model_name and variants:
            variant_list = ", ".join(variants)
            return (
                f"The {brand_name} {model_name} has multiple tyre options depending on the variant. "
                f"Which variant do you have? For example: {variant_list}."
            )
        elif brand_name and model_name:
            return f"The {brand_name} {model_name} has multiple variants with different tyre sizes. Could you tell me your specific variant?"
        return "This vehicle has multiple variants with different tyre sizes. Could you tell me your specific variant?"

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
        best_overlap = max(float(row.get("token_overlap", 0.0)) for row in top_window)
        best_similarity = max(float(row.get("similarity_score", 0.0)) for row in top_window)

        lexical_pass = best_overlap >= required_overlap
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

        logger.info(f"Starting recommendation workflow for query: '{query}'")

        # ── Two-phase search ──
        # Phase 1: search with the current query alone.  If the top result's
        #   model name appears in the current query tokens, the user is asking
        #   about a specific vehicle → use these results directly.
        # Phase 2 (follow-ups only): if the current query doesn't name a vehicle
        #   (e.g. "the SX one", "tell me for all versions"), re-search with up
        #   to the last 3 user messages prepended so the vehicle name from prior
        #   turns is included.
        # This prevents history contamination: "recommended tyre for baleno"
        # after a Verna conversation must NOT return Verna results.

        search_query = query  # default: current query only

        try:
            vehicle_rows = self.vector_search.search(query, k=20)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"Error during search: {e}"

        # Check whether the current query explicitly names the top vehicle model
        current_tokens = {t for t in self._tokenize(query) if not self._is_year_token(t)}
        top_model_tokens = set()
        if vehicle_rows:
            top_model_tokens = set(self._tokenize(str(vehicle_rows[0].get("vehicle-model", ""))))
        current_names_vehicle = bool(
            current_tokens and top_model_tokens and current_tokens.intersection(top_model_tokens)
        )

        # Check if the query is ambiguous across multiple brands
        # (e.g. "x3" matches both BMW X3 and a bad AUDI X3 record).
        matched_brands = set()
        for row in vehicle_rows[:10]:
            model_tokens = set(self._tokenize(str(row.get("vehicle-model", ""))))
            if current_tokens.intersection(model_tokens):
                matched_brands.add(str(row.get("vehicle-brand", "")).strip().lower())
        query_is_ambiguous = len(matched_brands) > 1

        if (not current_names_vehicle or query_is_ambiguous) and history:
            # Follow-up: re-search with history context for proper vehicle recall
            previous_user_msgs = [m["content"] for m in history if m["role"] == "user"]
            if previous_user_msgs:
                context_msgs = previous_user_msgs[-3:]
                search_query = f"{' '.join(context_msgs)} {query}"
                logger.info(f"Follow-up detected — re-searching with history: '{search_query}'")
                try:
                    vehicle_rows = self.vector_search.search(search_query, k=20)
                except Exception as e:
                    logger.error(f"History search failed: {e}")
        else:
            logger.info(f"Current query names vehicle directly — using Phase 1 results.")

        # Intent detection always uses the CURRENT user message (`query`), NOT the
        # combined `search_query`.  The combined string is only for vector retrieval —
        # using it for intent would bleed keywords from a previous turn into the current
        # intent classifier (e.g. "variants" from a prior listing question would keep
        # re-triggering the variant-listing branch on every follow-up).
        #
        # Response builders also receive `query` so model-matching uses the current
        # message; `vehicle_rows` already carry the right candidates from `search_query`.

        # Check brand-only ambiguity FIRST — before the context match gate —
        # so "what's the tyre for Audi" always gets a model clarification
        # regardless of whether the similarity score crosses the threshold.
        if vehicle_rows and self._query_is_brand_only_ambiguous(query, vehicle_rows):
            logger.info("Brand-only ambiguous query detected. Returning model clarification.")
            return self._get_brand_clarification(query, vehicle_rows)

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

    def recommend_stream(self, query: str, history: list[dict] | None = None):
        """
        Same as recommend() but yields response chunks for streaming.
        """
        # Handle greetings
        if self._is_greeting(query):
            yield "Hi! I'm your tyre recommendation assistant. Tell me your vehicle's make and model and I'll suggest the right tyres."
            return

        # Handle closings/thanks
        if self._is_thanks_or_closing(query) and not history:
            yield "You're welcome! Feel free to ask if you need tyre recommendations for any other vehicle."
            return

        logger.info(f"Starting recommendation workflow for query: '{query}'")

        search_query = query

        try:
            vehicle_rows = self.vector_search.search(query, k=20)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            yield f"Error during search: {e}"
            return

        current_tokens = {t for t in self._tokenize(query) if not self._is_year_token(t)}
        top_model_tokens = set()
        if vehicle_rows:
            top_model_tokens = set(self._tokenize(str(vehicle_rows[0].get("vehicle-model", ""))))
        current_names_vehicle = bool(
            current_tokens and top_model_tokens and current_tokens.intersection(top_model_tokens)
        )

        matched_brands = set()
        for row in vehicle_rows[:10]:
            model_tokens = set(self._tokenize(str(row.get("vehicle-model", ""))))
            if current_tokens.intersection(model_tokens):
                matched_brands.add(str(row.get("vehicle-brand", "")).strip().lower())
        query_is_ambiguous = len(matched_brands) > 1

        if (not current_names_vehicle or query_is_ambiguous) and history:
            previous_user_msgs = [m["content"] for m in history if m["role"] == "user"]
            if previous_user_msgs:
                context_msgs = previous_user_msgs[-3:]
                search_query = f"{' '.join(context_msgs)} {query}"
                logger.info(f"Follow-up detected — re-searching with history: '{search_query}'")
                try:
                    vehicle_rows = self.vector_search.search(search_query, k=20)
                except Exception as e:
                    logger.error(f"History search failed: {e}")
        else:
            logger.info(f"Current query names vehicle directly — using Phase 1 results.")

        if vehicle_rows and self._query_is_brand_only_ambiguous(query, vehicle_rows):
            logger.info("Brand-only ambiguous query detected. Returning model clarification.")
            yield self._get_brand_clarification(query, vehicle_rows)
            return

        has_match = self._has_strong_context_match(search_query, vehicle_rows)

        if not has_match:
            if history:
                logger.info("No strong context match but history present — routing to LLM for conversational reply.")
                try:
                    for chunk in self.response_generator._generate_conversational_stream(query, history):
                        yield chunk
                    return
                except Exception as e:
                    logger.error(f"Conversational fallback failed: {e}")
            logger.info("No sufficiently relevant context found. Returning helpful redirect.")
            yield "I can help you find the right tyres. Please share the make and model of your vehicle (e.g. Honda City, Maruti Swift, Toyota Fortuner)."
            return

        top_brand = str(vehicle_rows[0].get("vehicle-brand", "")).strip().lower()
        top_model = str(vehicle_rows[0].get("vehicle-model", "")).strip().lower()
        filtered_rows = [
            r for r in vehicle_rows
            if str(r.get("vehicle-brand", "")).strip().lower() == top_brand
            and str(r.get("vehicle-model", "")).strip().lower() == top_model
        ]
        if not filtered_rows:
            filtered_rows = vehicle_rows

        try:
            for chunk in self.response_generator.generate_stream(query, filtered_rows, history=history):
                yield chunk
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            yield f"Error creating recommendation: {e}"
