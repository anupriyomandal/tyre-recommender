from pathlib import Path
from src.search.vector_search import VectorSearch
from src.llm.response_generator import ResponseGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TyreRecommender:
    """
    Main recommender pipeline orchestrating search and LLM generation.
    """
    def __init__(self, index_path: Path | str, metadata_path: Path | str):
        self.vector_search = VectorSearch(index_path, metadata_path)
        self.response_generator = ResponseGenerator()

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

        if not vehicle_rows:
            return "No matching vehicles found for your query. Please try adjusting your search terms."

        # 3 & 4. Generate and return natural language answer
        try:
            answer = self.response_generator.generate(query, vehicle_rows, history=history)
            return answer
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"Error creating recommendation: {e}"
