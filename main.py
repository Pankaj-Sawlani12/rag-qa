from query import ask

def main():
    print("RAG Q&A — type a question and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not question:
            continue
        ask(question)
        print()

if __name__ == "__main__":
    main()
