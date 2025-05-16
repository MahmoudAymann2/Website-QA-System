import os
import pickle
import faiss
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import logging
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

retriever = None

def hybrid_search(query: str, hybrid: bool = True, top_k: int = 5) -> List[str]:
    global retriever
    if retriever is None:
        retriever = VectorStore()
        retriever.load_index()
    
    results = retriever.search(query, top_k=top_k, use_hybrid=hybrid)
    return [chunk for chunk, _ in results]

class VectorStore:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: str = "faiss_index.idx",
        meta_path: str = "faiss_metadata.pkl"
    ):
        """
        Initializes the VectorStore with FAISS and BM25 indices.
        """
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.text_chunks = []
        self.bm25_tokenized = []
        self.bm25 = None

    def load_chunks(self, pickle_file: str) -> List[str]:
        """
        Loads text chunks from a pickle file.
        
        Args:
            pickle_file (str): Path to the pickle file containing preprocessed chunks.
        
        Returns:
            List[str]: Loaded text chunks.
        """
        if not os.path.exists(pickle_file):
            raise FileNotFoundError(f"Pickle file not found: {pickle_file}")
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)
            self.text_chunks = data.get("chunks", [])
        return self.text_chunks

    def build_index(self, chunks: List[str]):
        """
        Builds FAISS and BM25 indices from text chunks.
        
        Args:
            chunks (List[str]): List of text chunks to index.
        """
        print("🔧 Encoding chunks...")
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        dim = embeddings.shape[1]

        # Build FAISS index
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings).astype("float32"))
        self.text_chunks = chunks

        # Save FAISS index and metadata
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump({"chunks": self.text_chunks}, f)

        print(f"✅ FAISS index saved to '{self.index_path}' and metadata to '{self.meta_path}'")

        # Build BM25 index
        self.bm25_tokenized = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(self.bm25_tokenized)

    def load_index(self):
        """
        Loads the FAISS index and BM25 index from disk.
        """
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found at '{self.index_path}'")
        self.index = faiss.read_index(self.index_path)

        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Metadata file not found at '{self.meta_path}'")
        with open(self.meta_path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                self.text_chunks = data.get("chunks", [])
            elif isinstance(data, list):
                self.text_chunks = data  # fallback if saved as a plain list
            else:
                raise ValueError("Unsupported metadata format in file.")

        # Load BM25 index
        self.bm25_tokenized = [chunk.lower().split() for chunk in self.text_chunks]
        self.bm25 = BM25Okapi(self.bm25_tokenized)


    def update_index(self, new_chunks: List[str]):
        """
        Updates the FAISS index with new chunks without rebuilding it entirely.
        """
        new_embeddings = self.model.encode(new_chunks, show_progress_bar=True)
        self.index.add(np.array(new_embeddings).astype("float32"))
        self.text_chunks.extend(new_chunks)

        # Save updated index and metadata
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump({"chunks": self.text_chunks}, f)

        print(f"✅ FAISS index updated with {len(new_chunks)} new chunks.")

    def search(self, query: str, top_k: int = 5, use_hybrid: bool = True) -> List[Tuple[str, float]]:
        """
        Searches for relevant chunks using FAISS and/or BM25.
        
        Args:
            query (str): User's question.
            top_k (int): Number of top results to return.
            use_hybrid (bool): Whether to use hybrid retrieval (FAISS + BM25).
        
        Returns:
            List[Tuple[str, float]]: Ranked list of (chunk, score) pairs.
        """
        if self.index is None or self.bm25 is None:
            raise RuntimeError("Index not loaded. Run 'load_index()' first.")

        # FAISS search
        query_embed = self.model.encode([query])
        D, I = self.index.search(np.array(query_embed).astype("float32"), top_k * 2)
        dense_results = [
            (self.text_chunks[i], 1 / (D[0][j] + 1e-5))  # Inverse L2 distance as similarity score
            for j, i in enumerate(I[0])
        ]

        # BM25 search
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)

        # Normalize scores
        scaler = MinMaxScaler()
        dense_scores = [score for _, score in dense_results]
        bm25_scores = list(scaler.fit_transform(np.array(bm25_scores).reshape(-1, 1)).flatten())
        dense_scores = list(scaler.fit_transform(np.array(dense_scores).reshape(-1, 1)).flatten())

        # Combine normalized scores
        combined = {}
        for text, dense_score, bm25_score in zip(self.text_chunks, dense_scores, bm25_scores):
            combined_score = 0.7 * dense_score + 0.3 * bm25_score
            combined[text] = combined_score

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


if __name__ == "__main__":
    try:
        retriever = VectorStore()

        # Step 1: Build or Load Index
        if not os.path.exists(retriever.index_path):
            print("📥 Loading preprocessed chunks...")
            chunks = retriever.load_chunks("preprocessed_data.pkl")
            retriever.build_index(chunks)
        else:
            print("📦 Loading existing FAISS + BM25 index...")
            retriever.load_index()

        # Step 2: Query
        user_question = "What will I find inside the Grand Egyptian Museum?"
        print(f"\n🔍 Query: \"{user_question}\"\n")
        top_results = retriever.search(user_question, top_k=5, use_hybrid=True)

        for i, (text, score) in enumerate(top_results, 1):
            print(f"{i}. [{score:.4f}] {text}\n")

    except Exception as e:
        print(f"🚨 Error: {e}")