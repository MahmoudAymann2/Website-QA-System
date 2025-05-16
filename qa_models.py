from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

# === EXTRACTIVE QA MODEL ===
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    tokenizer="deepset/roberta-base-squad2"
)

# === GENERATIVE QA MODEL (FLAN-T5-LARGE) ===
t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

# === RETRIEVAL MODEL FOR CONTEXT FILTERING ===
retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')


def retrieve_relevant_contexts(question: str, contexts: List[str], top_k: int = 2, similarity_threshold: float = 0.5) -> List[str]:
    """
    Selects the most semantically relevant contexts based on the question.
    
    Args:
        question (str): User question.
        contexts (List[str]): List of candidate contexts.
        top_k (int): Number of most relevant contexts to return.
        similarity_threshold (float): Minimum cosine similarity to consider a context relevant.

    Returns:
        List[str]: Top-K most relevant contexts (filtered by threshold).
    """
    if not contexts:
        return []

    # Encode all contexts and the question
    context_embeddings = retrieval_model.encode(contexts, convert_to_tensor=True)
    question_embedding = retrieval_model.encode(question, convert_to_tensor=True)

    # Compute cosine similarities
    scores = util.cos_sim(question_embedding, context_embeddings)[0]
    ranked_indices = scores.argsort(descending=True).tolist()

    # Filter out low-similarity contexts
    filtered_indices = [i for i in ranked_indices if scores[i] >= similarity_threshold]

    return [contexts[i] for i in filtered_indices[:top_k]]


def extract_answer(question: str, contexts: List[str], top_k: int = 3, score_threshold: float = 0.5) -> List[Dict[str, str]]:
    """
    Extracts answers from a list of contexts using a question.
    
    Args:
        question (str): The user's question.
        contexts (List[str]): Retrieved text chunks.
        top_k (int): Number of best answers to return.
        score_threshold (float): Minimum score to accept an answer.
    
    Returns:
        List[Dict[str, str]]: Ranked list of answers with metadata.
    """
    answers = []
    for i, context in enumerate(contexts):
        try:
            result = qa_pipeline(question=question, context=context)
            if isinstance(result, dict) and result["score"] >= score_threshold:
                result["context"] = context
                answers.append(result)
        except Exception as e:
            logging.error(f"Error during QA on chunk {i}: {e}")

    # Sort by score
    sorted_answers = sorted(answers, key=lambda x: x["score"], reverse=True)
    return sorted_answers[:top_k]


def fallback_response(question: str, combined_context: str) -> str:
    """
    Provides a fallback response using semantic similarity to find the most relevant sentence.
    """
    if not combined_context.strip():
        return "I'm sorry, I couldn't find an exact answer."

    # Split into sentences
    sentences = [s.strip() + "." for s in combined_context.split(".") if s.strip()]

    if not sentences:
        return "I'm sorry, I couldn't find an exact answer."

    # Find the sentence most similar to the question
    sentence_embeddings = retrieval_model.encode(sentences, convert_to_tensor=True)
    question_embedding = retrieval_model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, sentence_embeddings)[0]

    best_sentence = sentences[scores.argmax()]
    return f"I found some related information: {best_sentence}"


def generate_answer(question: str, contexts: List[str]) -> str:
    """
    Rewrites the question and generates a full-sentence answer using the FLAN-T5-Large model.
    
    Args:
        question (str): The user's question.
        contexts (List[str]): Retrieved text chunks.
    
    Returns:
        str: Generated answer.
    """
    logging.info(f"Raw Question: {question}")
    logging.info(f"Raw Contexts: {contexts}")

    # Filter contexts using semantic relevance
    relevant_contexts = retrieve_relevant_contexts(question, contexts)
    if not relevant_contexts:
        logging.info("No relevant context found.")
        return "I'm sorry, I don't have any information about that."

    combined_context = " ".join(relevant_contexts)[:3000]

    # === Step 1: Rewrite the Question ===
    rewrite_prompt = (
        f"You are a helpful assistant.\n"
        f"Rewrite the following question to make it more complete and natural:\n\n"
        f"Original Question: {question}\n"
        f"Rewritten Question:"
    )

    logging.info(f"T5 Rewrite Prompt: {rewrite_prompt}")

    rewrite_inputs = t5_tokenizer(
        rewrite_prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    rewrite_outputs = t5_model.generate(
        **rewrite_inputs,
        max_length=64,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    rewritten_question = t5_tokenizer.decode(rewrite_outputs[0], skip_special_tokens=True).strip()
    logging.info(f"Rewritten Question: {rewritten_question}")

    # === Step 2: Generate the Answer ===
    answer_prompt = (
        f"You are an expert Q&A assistant. "
        f"Please answer the question strictly based on the provided context below. "
        f"If the information is not present in the context, respond with: 'I'm sorry, I don't know.'\n\n"
        f"Context:\n{combined_context}\n\n"
        f"Question: {rewritten_question}\n"
        f"Answer:"
    )

    logging.info(f"T5 Answer Prompt: {answer_prompt}")

    answer_inputs = t5_tokenizer(
        answer_prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    answer_outputs = t5_model.generate(
        **answer_inputs,
        max_length=150,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True
    )

    generated_ids = answer_outputs.sequences[0]
    generated_answer = t5_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Confidence estimation from the first sequence
    if answer_outputs.sequences_scores:
        confidence = answer_outputs.sequences_scores[0].exp().item()
    else:
        confidence = 0.0

    logging.info(f"Generated Answer Confidence: {confidence:.4f}")

    # === Fallback Mechanism ===
    if len(generated_answer.split()) < 5 or "I'm sorry" in generated_answer.lower():
        generated_answer = fallback_response(question, combined_context)

    logging.info(f"Generated Final Answer: {generated_answer}")

    return generated_answer


if __name__ == "__main__":
    # Sample test
    question = "Who is Fifi Abdo?"
    example_contexts = [
        "Fifi Abdo is an Egyptian belly dancer and actress known for her performances in Cairo cabaret shows.",
        "The Grand Egyptian Museum will house over 100,000 artifacts.",
        "It will display the treasures of Tutankhamun and other ancient Egyptian relics.",
        "The museum is located near the Giza Pyramids and covers an area of 500,000 square meters."
    ]

    # Test extractive QA
    top_answers = extract_answer(question, example_contexts, top_k=3)
    print("\n🧠 Top Extractive Answers:")
    for i, ans in enumerate(top_answers, 1):
        print(f"{i}. Answer: {ans['answer']} (Score: {ans['score']:.4f})\n   From: {ans['context']}")

    # Test generative QA
    generated_answer = generate_answer(question, example_contexts)
    print("\n🤖 Generated Answer:")
    print(generated_answer)