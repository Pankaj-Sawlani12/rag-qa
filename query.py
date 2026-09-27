import os
import sys

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import config

PROMPT_TEMPLATE = """Use the following context to answer the question in
full, well-explained sentences (at least 2-3 sentences). Do not answer
with just a short phrase. If the answer is not contained in the context,
say you don't know — do not make up an answer.

Context:
{context}

Question: {question}

Detailed answer:"""


def load_vectorstore():
    if not os.path.isdir(config.INDEX_DIR):
        print(f"No FAISS index found at '{config.INDEX_DIR}'.")
        print("Run 'python ingest.py' first to build the index from your PDFs.")
        sys.exit(1)

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.load_local(
        config.INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,  # safe: it's our own local index
    )
    return vectorstore


def load_llm():

    from langchain_huggingface import HuggingFacePipeline
    from transformers import pipeline

    print(f"Loading local generation model: {config.GENERATION_MODEL_NAME}")
    print("(first run downloads the model; this may take a minute)")
    pipe = pipeline(
        "text2text-generation",
        model=config.GENERATION_MODEL_NAME,
        max_new_tokens=300,
        min_new_tokens=64,
        num_beams=4,
        truncation=True,
    )
    return HuggingFacePipeline(pipeline=pipe)


def ask(question: str):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})
    llm = load_llm()

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    print(f"\nQuestion: {question}\n")
    print("Retrieving relevant chunks and generating answer...\n")
    result = qa_chain.invoke({"query": question})

    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result["result"].strip())

    print("\n" + "=" * 60)
    print("SOURCES USED")
    print("=" * 60)
    for i, doc in enumerate(result["source_documents"], start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        snippet = doc.page_content[:150].replace("\n", " ")
        print(f"[{i}] {os.path.basename(source)} (page {page}): {snippet}...")


def main():
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question here"')
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    ask(question)


if __name__ == "__main__":
    main()
