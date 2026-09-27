import os
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def load_pdfs(data_dir: str):

    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in '{data_dir}'.")
        print("Add some .pdf files there and re-run this script.")
        sys.exit(1)

    all_docs = []
    for filename in pdf_files:
        path = os.path.join(data_dir, filename)
        print(f"Loading: {filename}")
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)

    print(f"Loaded {len(all_docs)} pages from {len(pdf_files)} PDF(s).")
    return all_docs


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks "
          f"(chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}).")
    return chunks


def build_index(chunks):
    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

    print("Embedding chunks and building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(config.INDEX_DIR, exist_ok=True)
    vectorstore.save_local(config.INDEX_DIR)
    print(f"FAISS index saved to: {config.INDEX_DIR}")


def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    documents = load_pdfs(config.DATA_DIR)
    chunks = split_documents(documents)
    build_index(chunks)
    print("\nIngestion complete.")


if __name__ == "__main__":
    main()
