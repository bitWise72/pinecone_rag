# project_root/vector_db/embedder.py
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME, PINECONE_DIMENSION

class Embedder:
    def __init__(self):
        """Initializes the sentence transformer model."""
        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded.")
            # Optional: Verify the dimension matches the configured dimension
            dummy_embedding = self.model.encode("test").tolist()
            if len(dummy_embedding) != PINECONE_DIMENSION:
                 print(f"WARNING: Configured PINECONE_DIMENSION ({PINECONE_DIMENSION}) does not match model output dimension ({len(dummy_embedding)}). Please update config.py.")

        except Exception as e:
            print(f"Error loading embedding model '{EMBEDDING_MODEL_NAME}': {e}")
            self.model = None

    def encode(self, text):
        """Encodes a given text string into a vector embedding."""
        if not self.model:
            raise RuntimeError("Embedding model not loaded. Cannot encode text.")
        return self.model.encode(text).tolist() # Return as list for Pinecone

# Create a singleton instance of the embedder
embedder = Embedder()