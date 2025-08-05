from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import google.generativeai as genai

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# Set API key directly
API_KEY = "AIzaSyAhxI1iwIiOdO2nlJF4q0dleRbELx0-CGY"
genai.configure(api_key=API_KEY)

# Global variables for model and embeddings
model = None
chunks = []
metadata = []
chunk_embeddings = []

def load_chunks():
    global chunks, metadata
    chunks = []
    metadata = []
    chunks_dir = "./chunks"
    
    print("📂 Loading all chunks from file system...")
    for filename in os.listdir(chunks_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(chunks_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:  # Only add non-empty chunks
                    chunks.append(content)
                    metadata.append({"source": filename})
    
    print(f"✅ Loaded {len(chunks)} chunks from file system")
    return chunks, metadata

def load_model():
    global model
    print("🔄 Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded successfully")
    return model

def generate_embeddings():
    global chunks, chunk_embeddings
    print(f"🔄 Embedding {len(chunks)} chunks... (this may take a moment)")
    chunk_embeddings = model.encode(chunks)
    print("✅ All chunks embedded successfully")
    return chunk_embeddings

@app.route('/')
def index():
    return app.send_static_file('policy_assistant_ui.html')

@app.route('/api/query', methods=['POST'])
def query():
    data = request.json
    user_query = data.get('query', '')
    
    if not user_query:
        return jsonify({"error": "Query is required"}), 400
    
    try:
        # Embed query
        query_embedding = model.encode(user_query)
        
        # Calculate similarity scores
        similarities = []
        for chunk_embedding in chunk_embeddings:
            similarity = np.dot(query_embedding, chunk_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding))
            similarities.append(similarity)
        
        # Get top 5 most similar chunks
        top_k = 5
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_chunks = [chunks[i] for i in top_indices]
        top_metadata = [metadata[i] for i in top_indices]
        top_scores = [similarities[i] for i in top_indices]
        
        formatted_results = []
        for i, (chunk, meta, score) in enumerate(zip(top_chunks, top_metadata, top_scores)):
            formatted_results.append({
                "rank": i + 1,
                "score": float(score),
                "source": meta["source"],
                "content": chunk[:300] + "..." if len(chunk) > 300 else chunk
            })
        
        # Generate AI response
        try:
            # Prepare content for the prompt (combine top chunks)
            top_chunks_content = ""
            for i, (chunk, meta) in enumerate(zip(top_chunks, top_metadata)):
                top_chunks_content += f"Clause {i+1} from {meta['source']}:\n{chunk}\n\n"
            
            prompt = f"""
            Query from user: {user_query}
            
            Relevant insurance policy clauses:
            
            {top_chunks_content}
            
            Based on the above policy clauses, please provide an answer in the following JSON format:
            {{
              "decision": "Approved/Rejected/Partial/Need more info",
              "justification": "Detailed explanation based on the policy clauses above. Include clause numbers when referring to specific clauses.",
              "amount": "Amount covered if applicable, otherwise 'N/A'",
              "next_steps": "What the user should do next"
            }}
            
            Only return the JSON object, nothing else.
            """
            
            model_gemini = genai.GenerativeModel("gemini-1.5-flash")
            response = model_gemini.generate_content(prompt)
            ai_response = response.text.strip()
            
            # Try to parse the response as JSON
            try:
                parsed_response = json.loads(ai_response.replace('```json', '').replace('```', '').strip())
            except json.JSONDecodeError:
                # If the response is not valid JSON, return it as a string
                parsed_response = {
                    "decision": "Need more info",
                    "justification": "The system could not parse the response. Here's the raw output:\n\n" + ai_response,
                    "amount": "N/A",
                    "next_steps": "Please try rephrasing your query or contact customer support."
                }
            
            return jsonify({
                "success": True,
                "matches": formatted_results,
                "ai_response": parsed_response
            })
        
        except Exception as e:
            print(f"⚠️ Error with Gemini API: {e}")
            return jsonify({
                "success": True,
                "matches": formatted_results,
                "ai_response": {
                    "decision": "Need more info",
                    "justification": "Based on the top matching policy clauses, additional information is needed about your specific situation to determine eligibility.",
                    "amount": "N/A (depends on specific policy terms)",
                    "next_steps": "Please contact your insurance provider with your policy number and details of your situation."
                }
            })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Load chunks and model on startup
    chunks, metadata = load_chunks()
    model = load_model()
    chunk_embeddings = generate_embeddings()
    
    # Start the Flask app
    app.run(debug=True, port=5000)