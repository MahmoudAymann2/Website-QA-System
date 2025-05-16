import logging
import re
from typing import List
from sentence_transformers import SentenceTransformer
import faiss

# === LOGGING UTILITIES ===
def configure_logging(level=logging.INFO):
    """
    Configures logging for the application.
    
    Args:
        level (int): Logging level (default: INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.info("Logging configured successfully.")

# === TEXT PREPROCESSING UTILITIES ===
def clean_text(text: str) -> str:
    """
    Cleans text by removing duplicate words, fixing punctuation, and normalizing whitespace.
    
    Args:
        text (str): Input text to clean.
    
    Returns:
        str: Cleaned text.
    """
    # Remove duplicate words (e.g., "an an" -> "an")
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)
    
    # Add spaces after punctuation if missing
    text = re.sub(r'(?<=[a-zA-Z])\.(?=\s*[A-Z])', '. ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def combine_chunks(chunks: list, max_length: int = 500) -> str:
    """
    Combines multiple chunks into a single context string, not exceeding the max_length characters.
    
    Args:
        chunks (list): List of text chunks to combine.
        max_length (int): Maximum length of the combined context.
    
    Returns:
        str: Combined context string.
    """
    combined = ""
    for chunk in chunks:
        if len(combined) + len(chunk) <= max_length:
            combined += " " + chunk.strip()
        else:
            break
    return combined.strip()

# === FAISS INDEX UTILITIES ===
def update_faiss_index(index, new_contexts: List[str], embedder: SentenceTransformer):
    """
    Updates a FAISS index with new contexts.
    
    Args:
        index: The FAISS index object.
        new_contexts (List[str]): List of new contexts to add.
        embedder (SentenceTransformer): Model for encoding new contexts.
    
    Returns:
        None
    """
    if not new_contexts:
        raise ValueError("No new contexts provided.")
    
    # Encode new contexts
    new_embeddings = embedder.encode(new_contexts, convert_to_numpy=True)
    
    # Add to FAISS index
    index.add(new_embeddings)
    
    logging.info(f"Added {len(new_contexts)} new contexts to the FAISS index.")