<h1>🤖 Website QA System</h1>

<p>A question-answering system that extracts and generates answers from any website's content using a hybrid search and transformer-based models.</p>

<hr>

<h2>🚀 Features</h2>
<ul>
  <li>Extractive and generative question answering</li>
  <li>Hybrid document retrieval (BM25 + FAISS)</li>
  <li>Built with FastAPI for easy API access</li>
  <li>Custom embeddings and local vector store</li>
</ul>

<hr>

<h2>🧠 Tech Stack & Explanation</h2>

<h3>1. 📄 Retrieval Methods</h3>
<ul>
  <li><strong>BM25</strong>: Sparse keyword-based retrieval using <code>rank_bm25</code>.</li>
  <li><strong>FAISS</strong>: Dense semantic retrieval using vector similarity from Sentence Transformers.</li>
  <li><strong>Hybrid</strong>: Combines both BM25 and FAISS for improved relevance.</li>
</ul>

<h3>2. 🤖 Answering Modes</h3>
<ul>
  <li><strong>Extractive</strong>: Finds exact answer spans using <code>deepset/roberta-base-squad2</code>.</li>
  <li><strong>Generative</strong>: Synthesizes answers with <code>google/flan-t5-large</code>.</li>
  <li><strong>Fallback</strong>: If no direct answer, selects most relevant sentence using semantic similarity.</li>
</ul>

<h3>3. 🧰 Libraries</h3>
<ul>
  <li><code>FastAPI</code> – REST API backend</li>
  <li><code>transformers</code>, <code>sentence-transformers</code> – QA models and embedding</li>
  <li><code>faiss</code> – Vector search engine</li>
  <li><code>rank_bm25</code> – Keyword matching engine</li>
</ul>

<hr>

<h2>📁 Project Structure</h2>

<pre>
website_qa_model/
│
├── <strong>data_collection.py</strong>       → Scrapes and extracts raw text from web pages
├── <strong>preprocessing.py</strong>         → Cleans, splits, and deduplicates raw text into chunks
├── <strong>retrieval.py</strong>             → Builds and manages the FAISS + BM25 hybrid index
├── <strong>qa_models.py</strong>             → Contains logic for extractive/generative QA using Transformers
├── <strong>api.py</strong>                   → FastAPI backend exposing endpoints for question answering
</pre>

<h3>📄 File Descriptions</h3>
<ul>
  <li><strong>data_collection.py</strong>: Uses requests and BeautifulSoup to crawl and extract HTML content from a given site.</li>
  <li><strong>preprocessing.py</strong>: Tokenizes paragraphs, removes duplicates, and prepares clean input chunks for indexing.</li>
  <li><strong>retrieval.py</strong>: Encodes all chunks into dense vectors and indexes them with FAISS. Also supports sparse BM25 search.</li>
  <li><strong>qa_models.py</strong>: Loads pre-trained QA models from Hugging Face. Performs extractive and generative QA based on context.</li>
  <li><strong>api.py</strong>: Exposes API routes like <code>/ask</code> and <code>/update</code> to interact with the QA system via HTTP.</li>
</ul>

<hr>

<h2>🛠️ Installation</h2>

<pre>
git clone https://github.com/yourusername/website-qa-model.git
cd website-qa-model
pip install -r requirements.txt
</pre>

<h3>👉 Environment Notes</h3>
<ul>
  <li>Python 3.8+</li>
  <li>You'll need a GPU for efficient generation (recommended)</li>
</ul>

<hr>

<h2>🔌 API Usage</h2>

<h3>POST /ask</h3>
<pre>
POST http://localhost:8000/ask
</pre>

<strong>Body:</strong>
<pre>
{
  "question": "What will I find inside the Grand Egyptian Museum?"
}
</pre>

<strong>Sample Output:</strong>
<pre>
{
  "question": "What will I find inside the Grand Egyptian Museum?",
  "answers": [
    {
      "answer": "Inside the Grand Egyptian Museum, you will find over 100,000 ancient Egyptian artifacts including the treasures of Tutankhamun...",
      "context": "The Grand Egyptian Museum will house over 100,000 artifacts. It will display the treasures of Tutankhamun and other ancient Egyptian relics."
    }
  ]
}
</pre>

<hr>

<h2>🌐 Demo Example</h2>
<p>This project was used to analyze the website: <a href="https://grandegyptianmuseum.org" target="_blank">grandegyptianmuseum.org</a></p>

<hr>

<h2>📦 Deployment (Optional)</h2>
<ul>
  <li>Can be deployed on <strong>Render</strong>, <strong>Railway</strong>, or <strong>Docker</strong></li>
  <li>Just point the container/entry to <code>api.py</code> with <code>uvicorn api:app --host 0.0.0.0 --port 8000</code></li>
</ul>

<hr>

<h2>🤝 Contributing</h2>
<p>Feel free to fork this repo, file issues, or open PRs!</p>

<hr>

<h2>📜 License</h2>
<p>MIT License</p>
