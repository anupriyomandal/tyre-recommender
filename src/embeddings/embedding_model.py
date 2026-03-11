import openai
import numpy as np
from src.config import EMBEDDING_MODEL_NAME, OPENAI_API_KEY
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize OpenAI client
if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here":
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None
    logger.warning("OPENAI_API_KEY is not set. Embedding generation will fail.")


class EmbeddingModel:
    """
    OpenAI model for generating embeddings.
    """
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name

    def load_model(self):
        if not client:
            raise ValueError("OpenAI API key is not configured.")

    def encode_documents(self, documents: list[str]) -> np.ndarray:
        """
        Encode a list of text documents into a numpy array (float32).
        """
        self.load_model()
        logger.info(f"Encoding {len(documents)} documents using OpenAI {self.model_name}...")
        
        # OpenAI has a limit on batch size, but for ~2000 documents it should be fine depending on token count.
        # To be safe, we will batch them in chunks of 500
        all_embeddings = []
        batch_size = 500
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            response = client.embeddings.create(input=batch, model=self.model_name)
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
            
        logger.info("Done generating embeddings.")
        return np.array(all_embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string.
        """
        self.load_model()
        response = client.embeddings.create(input=[query], model=self.model_name)
        embedding = response.data[0].embedding
        return np.array([embedding], dtype=np.float32)
