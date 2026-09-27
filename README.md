```bash
# 1. Add PDF files to data/
python ingest.py

# User Query
python query.py "your question here"

## Setup

pip install -r requirements.txt
```

## Usage

```bash
# 1. Add PDF files to data/
# 2. Build the index
python ingest.py

# 3. Ask a question
python query.py "your question here"

# Or start an interactive session
python main.py
```

## WORKING
*Ingestion* - PDFs are loaded, split into overlapping chunks, and converted into vector embeddings.

*Indexing* - Embeddings are stored in a FAISS index for fast similarity search.

*Retrieval* - A query is embedded and matched against the index to find the most relevant chunks.

*Generation* - Retrieved chunks are passed as context to a language model, which produces an answer grounded in the source documents.