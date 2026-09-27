import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

GENERATION_MODEL_NAME = "google/flan-t5-large"

CHUNK_SIZE = 350
CHUNK_OVERLAP = 40
TOP_K = 3

OPENAI_MODEL_NAME = "gpt-4o-mini"
