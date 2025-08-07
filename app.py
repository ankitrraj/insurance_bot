from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import json
import os
import re
import logging
import random
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure environment
DEBUG = True
FLASK_ENV = 'development'
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5000", "http://127.0.0.1:5000", "https://insurance-bot-49ae.onrender.com"],
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

@app.route('/api/test')
def test_api():
    try:
        return jsonify({
            "status": "API working", 
            "chunks_loaded": len(chunks),
            "metadata_loaded": len(metadata),
            "timestamp": str(datetime.now()) if 'datetime' in globals() else "N/A"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/debug')
def debug_info():
    try:
        chunks_dir = "./chunks"
        chunks_exist = os.path.exists(chunks_dir)
        chunks_files = []
        if chunks_exist:
            chunks_files = [f for f in os.listdir(chunks_dir) if f.endswith('.txt')]
        
        return jsonify({
            "chunks_directory_exists": chunks_exist,
            "chunks_files_count": len(chunks_files),
            "chunks_loaded_in_memory": len(chunks),
            "metadata_loaded_in_memory": len(metadata),
            "sample_files": chunks_files[:5] if chunks_files else []
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

# Global variables for data
chunks = []
metadata = []

def initialize_app():
    """Initialize the app with data"""
    global chunks, metadata
    
    try:
        # Load all chunks from the chunks directory
        logger.info("📂 Loading all chunks from file system...")
        chunks = []
        metadata = []
        chunks_dir = "./chunks"
        
        if not os.path.exists(chunks_dir):
            logger.error(f"Chunks directory {chunks_dir} not found!")
            return False
            
        for filename in os.listdir(chunks_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(chunks_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            chunks.append(content)
                            metadata.append({"source": filename})
                except Exception as e:
                    logger.warning(f"Error reading {filename}: {e}")
                    continue
                    
        logger.info(f"✅ Loaded {len(chunks)} chunks from file system")
        
        if len(chunks) == 0:
            logger.error("No chunks loaded! Check if chunks directory has .txt files")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error in initialization: {e}")
        return False

def simple_similarity_search(query, chunks, top_k=5):
    """Simple keyword-based similarity search"""
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    scores = []
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        chunk_words = set(chunk_lower.split())
        
        # Calculate simple word overlap
        common_words = query_words.intersection(chunk_words)
        score = len(common_words) / max(len(query_words), 1)
        
        # Bonus for exact phrase matches
        if query_lower in chunk_lower:
            score += 0.5
            
        scores.append((score, i))
    
    # Sort by score and return top_k
    scores.sort(reverse=True)
    return scores[:top_k]

@app.route('/api/query', methods=['POST'])
def query():
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400
        query = data['query']

        # Debug logging
        logger.info(f"Received query: {query}")
        logger.info(f"Chunks loaded: {len(chunks)}")
        logger.info(f"Metadata loaded: {len(metadata)}")

        if len(chunks) == 0:
            logger.error("No chunks loaded! Trying to reinitialize...")
            if not initialize_app():
                return jsonify({"error": "Failed to load data"}), 500
            if len(chunks) == 0:
                return jsonify({"error": "No data available"}), 500

        # Get top chunks using simple similarity
        top_results = simple_similarity_search(query, chunks, top_k=5)
        
        # Prepare context for the prompt
        context = ""
        for i, (score, idx) in enumerate(top_results):
            if score > 0:  # Only include relevant chunks
                context += f"Clause {i+1} from {metadata[idx]['source']}:\n{chunks[idx]}\n\n"

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
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                logger.info(f"🔄 Attempt {attempt + 1} with new API key")
                gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                response = gemini_model.generate_content(prompt)
                ai_response_text = response.text.strip()
                ai_response = ai_response_text
                logger.info("✅ Successfully got Gemini response")
                break
            except Exception as e:
                logger.error(f"❌ Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    ai_response = "Sorry, I could not process your request. Please try again or contact support."

        # Prepare matches for frontend
        matches = []
        for score, idx in top_results:
            if score > 0:  # Only include relevant matches
                matches.append({
                    "content": chunks[idx],
                    "score": float(score),
                    "source": metadata[idx]["source"]
                })

        return jsonify({
            "matches": matches,
            "ai_response": ai_response
        })
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize the app
    if initialize_app():
        logger.info("🚀 Starting Flask app...")
        port = int(os.environ.get('PORT', 5000))
        app.run(debug=True, port=port)
    else:
        logger.error("❌ Failed to initialize app. Check logs above.")
        exit(1)