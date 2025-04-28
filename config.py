# project_root/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Database Configuration ---
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME")

# --- Pinecone Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

# --- Pinecone Index Dimension ---
# IMPORTANT: This MUST match the output dimension of your chosen embedding model.
# For 'all-MiniLM-L6-v2', the dimension is 384.
# You will need to change this if you use a different model (e.g., OpenAI ada-002 is 1536).
PINECONE_DIMENSION = 768 


# Basic validation (optional but recommended)
if not all([MONGO_URI, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, EMBEDDING_MODEL_NAME, MONGO_DB_NAME, MONGO_COLLECTION_NAME]):
    raise ValueError("Missing one or more required environment variables. Check your .env file.")

print("Configuration loaded successfully.")