# main.py

import os
import argparse
from data_collection import collect_data
from preprocessing import preprocess_data
from retrieval import VectorStore
from qa_models import extract_answer


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Website QA System")
    parser.add_argument(
        "--action",
        choices=["collect", "preprocess", "build-index", "query"],
        required=True,
        help="Action to perform: 'collect', 'preprocess', 'build-index', or 'query'."
    )
    return parser.parse_args()


def main():
    try:
        args = parse_args()

        # Step 1: Collect raw data
        if args.action == "collect":
            print("Collecting raw data...")
            raw_data_file = "raw_data.txt"
            collect_data(output_file=raw_data_file)
            print("Raw data collected successfully.")

        # Step 2: Preprocess data
        elif args.action == "preprocess":
            print("Preprocessing data...")
            raw_data_file = "raw_data.txt"
            preprocessed_data_file = "preprocessed_data.pkl"
            if not os.path.exists(raw_data_file):
                raise FileNotFoundError(f"Raw data file not found: {raw_data_file}")
            preprocess_data(input_file=raw_data_file, output_file=preprocessed_data_file)
            print("Data preprocessed successfully.")

        # Step 3: Build FAISS index
        elif args.action == "build-index":
            print("Building FAISS index...")
            preprocessed_data_file = "preprocessed_data.pkl"
            if not os.path.exists(preprocessed_data_file):
                raise FileNotFoundError(f"Preprocessed data file not found: {preprocessed_data_file}")
            retriever = VectorStore()
            chunks = retriever.load_chunks(preprocessed_data_file)
            retriever.build_index(chunks)
            print("FAISS index built successfully.")

        # Step 4: Perform question-answering
        elif args.action == "query":
            print("Starting QA system...")
            retriever = VectorStore()

            # Load FAISS index
            if not os.path.exists(retriever.index_path):
                raise RuntimeError("FAISS index not found. Please build the index first.")
            retriever.load_index()

            while True:
                user_question = input("\nEnter your question (or type 'exit' to quit): ")
                if user_question.lower() == "exit":
                    print("Exiting...")
                    break

                if not user_question.strip():
                    print("Query cannot be empty.")
                    continue

                print(f"\n🔍 Searching for: \"{user_question}\"")
                top_results = retriever.search(user_question, top_k=5)

                if not top_results:
                    print("No relevant chunks found. Please try rephrasing your question.")
                    continue

                # Extract answers
                top_chunks = [chunk for chunk, _ in top_results]
                top_answers = extract_answer(user_question, top_chunks, top_k=2)

                print("\n🧠 Top Extracted Answers:")
                if not top_answers:
                    print("No answer found. Please try rephrasing your question or providing more context.")
                else:
                    for i, ans in enumerate(top_answers, 1):
                        print(f"{i}. Answer: {ans['answer']}")
                        print(f"   Confidence: {ans['score']:.4f}")
                        print(f"   Context: {ans['context'][:100]}...\n")

    except Exception as e:
        print(f"🚨 Error: {e}")


if __name__ == "__main__":
    main()