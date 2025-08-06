from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import numpy as np
import json
import os
import re

app = Flask(__name__)

# Configure environment
DEBUG = True
FLASK_ENV = 'development'
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5000", "http://127.0.0.1:5000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({"status": "ok", "message": "Server is running"})

# Configure API keys with rotation
API_KEYS = [
    "AIzaSyA7cEcHd2tQULZqllNGEQP5m3NcZLA-NqI",
    "AIzaSyCxuW9VMKVkv5USDlOqIzEDqAiouvuU8I0",
    "AIzaSyCasXllJ1gW42PDP7EIrPVBVB9LVSI1YK8"
]
current_key_index = 0

def get_next_api_key():
    global current_key_index
    key = API_KEYS[current_key_index]
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    return key

# Load all chunks from the chunks directory (from hackrx_full_demo.py)
print("📂 Loading all chunks from file system...")
chunks = []
metadata = []
chunks_dir = "./chunks"
for filename in os.listdir(chunks_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(chunks_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                chunks.append(content)
                metadata.append({"source": filename})
print(f"✅ Loaded {len(chunks)} chunks from file system")

# Load embedding model
print("🔄 Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedding model loaded")

# Embed all chunks
print(f"🔄 Embedding {len(chunks)} chunks... (this may take a moment)")
chunk_embeddings = embedding_model.encode(chunks)
print("✅ All chunks embedded successfully")

@app.route('/api/query', methods=['POST'])
def query():
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400
        query = data['query']

        # Embed query
        query_embedding = embedding_model.encode(query)

        # Calculate similarities
        similarities = []
        for chunk_embedding in chunk_embeddings:
            similarity = float(np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            ))
            similarities.append(similarity)

        # Get top 5 most similar chunks
        top_k = 5
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_chunks = [chunks[i] for i in top_indices]
        top_metadata = [metadata[i] for i in top_indices]
        top_scores = [similarities[i] for i in top_indices]

        # Prepare context for the prompt (combine top chunks)
        context = ""
        for i, (chunk, meta) in enumerate(zip(top_chunks, top_metadata)):
            context += f"Clause {i+1} from {meta['source']}:\n{chunk}\n\n"

        prompt = f"""
You are an expert insurance policy assistant. Follow this 3-step process to answer the user's query.

1. **Think Step** (simulate internal reasoning): Briefly describe what you're looking for in the policy document, and what criteria will help decide the answer.

2. **Explain to User**: Give a clear explanation in simple, user-friendly language based on the policy details provided.

3. **Structured Output**: Return a final structured answer in JSON format as shown.

---
Insurance Policy Context:
{context}

---
User Question:
{query}

---
Expected Final Output Format:
{{
  "query": "...",
  "decision": "Yes/No/Partial/Need more info",
  "justification": "Based on exact policy section/coverage clause",
  "amount": "Coverage amount if applicable",
  "source": "Policy clause or section",
  "next_steps": "What user should do now"
}}

Respond in this exact format:

🧠 **Thinking**:
...

💬 **Explanation**:
...

📦 **JSON**:
{{ ... }}
"""

        # Generate response with retry logic and API key rotation
        max_retries = 3
        ai_response = None
        for attempt in range(max_retries):
            try:
                api_key = get_next_api_key()
                genai.configure(api_key=api_key)
                print(f"🔄 Attempt {attempt + 1} with new API key")
                gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                response = gemini_model.generate_content(prompt)
                ai_response_text = response.text.strip()
                ai_response = ai_response_text
                print("✅ Successfully got Gemini response")
                break
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    ai_response = "Sorry, I could not process your request. Please try again or contact support."

        # Prepare matches for frontend
        matches = []
        for i, idx in enumerate(top_indices):
            matches.append({
                "content": chunks[idx],
                "score": float(similarities[idx]),
                "source": metadata[idx]["source"]
            })

        return jsonify({
            "matches": matches,
            "ai_response": ai_response
        })
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)