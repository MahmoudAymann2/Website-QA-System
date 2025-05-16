from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, validator
import os
import pickle
import numpy as np
import logging
from typing import List, Dict

from retrieval import VectorStore
from qa_models import extract_answer, generate_answer
from sentence_transformers import SentenceTransformer
import faiss

# Configure logging
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

# Initialize FastAPI app
app = FastAPI(
    title="Website QA System API",
    description="A REST API for answering questions based on website content using extractive and generative QA.",
    version="1.2.0"
)

# Load embedder globally
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize VectorStore
retriever = VectorStore()

@app.on_event("startup")
async def startup_event():
    """
    Load the FAISS + BM25 index and preprocessed data when the API starts.
    """
    if not os.path.exists(retriever.index_path):
        raise RuntimeError("FAISS index not found. Please build the index first.")
    retriever.load_index()
    logging.info("✅ FAISS + BM25 index loaded successfully.")

# ==== Pydantic Models ====

class QueryRequest(BaseModel):
    question: str

    @validator("question")
    def validate_question(cls, value):
        if len(value.strip()) < 3:
            raise ValueError("Question must be at least 3 characters long.")
        if not any(char.isalpha() for char in value):
            raise ValueError("Question must contain alphabetic characters.")
        return value.strip()

class UpdateRequest(BaseModel):
    new_contexts: List[str]

# ==== Utility ====

def combine_chunks(chunks: List[str], max_length: int = 1000) -> str:
    """
    Combines chunks into a readable format with a maximum length.
    """
    combined = ""
    for chunk in chunks:
        if len(combined) + len(chunk) + 1 > max_length:
            break
        combined += f"- {chunk}\n"
    return combined.strip()

# ==== Main QA Endpoint ====

@app.post("/ask/", response_model=dict)
async def ask(
    request: QueryRequest,
    hybrid: bool = Query(True, description="Use hybrid retrieval (BM25 + FAISS)?"),
    mode: str = Query("generative", description="'extractive', 'generative', or 'hybrid'")
):
    """
    Answer a user's question using extractive, generative, or hybrid QA.
    """
    user_question = request.question.strip()
    logging.info(f"🔍 Question: \"{user_question}\" | Mode: {mode} | Hybrid: {hybrid}")

    top_results = retriever.search(user_question, top_k=10, use_hybrid=hybrid)

    if not top_results:
        return {
            "question": user_question,
            "answers": [],
            "message": "No relevant chunks found. Try rephrasing your question."
        }

    top_chunks = [chunk for chunk, _ in top_results]

    if mode.lower() == "generative":
        combined_context = combine_chunks(top_chunks, max_length=1000)
        generated = generate_answer(user_question, top_chunks)
        return {
            "question": user_question,
            "answers": [{
                "answer": generated,
                "context": combined_context[:100] + "..."
            }]
        }

    elif mode.lower() == "extractive":
        top_answers = extract_answer(user_question, top_chunks, top_k=3)
        return {
            "question": user_question,
            "answers": [
                {
                    "answer": ans["answer"],
                    "confidence": round(ans["score"], 4),
                    "context": ans["context"][:100] + "..."
                }
                for ans in top_answers
            ]
        }

    elif mode.lower() == "hybrid":
        top_answers = extract_answer(user_question, top_chunks, top_k=3)
        generated = generate_answer(user_question, top_chunks)
        combined_context = combine_chunks(top_chunks, max_length=1000)
        return {
            "question": user_question,
            "answers": [
                {
                    "answer": ans["answer"],
                    "confidence": round(ans["score"], 4),
                    "context": ans["context"][:100] + "...",
                    "type": "extractive"
                }
                for ans in top_answers
            ] + [{
                "answer": generated,
                "context": combined_context[:100] + "...",
                "type": "generative"
            }]
        }

    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Choose 'extractive', 'generative', or 'hybrid'.")

# ==== Update Context Index Endpoint ====

@app.post("/update/")
async def update_index(request: UpdateRequest):
    """
    Updates the FAISS index with new contexts.
    """
    new_contexts = request.new_contexts
    if not new_contexts:
        raise HTTPException(status_code=400, detail="No new contexts provided.")

    embeddings = embedder.encode(new_contexts, show_progress_bar=True)
    retriever.index.add(np.array(embeddings).astype("float32"))
    retriever.text_chunks.extend(new_contexts)

    # Save updated index and metadata
    faiss.write_index(retriever.index, retriever.index_path)
    with open(retriever.meta_path, "wb") as f:
        pickle.dump({"chunks": retriever.text_chunks}, f)

    return {"message": f"Index updated successfully with {len(new_contexts)} new chunks."}
