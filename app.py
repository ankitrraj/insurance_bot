from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import numpy as np
import json
import os

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

# Initialize models
print("🔄 Loading models...")
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Embedding model loaded")
    
    # Configure Gemini model with proper settings
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 40,
    }
    gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest", generation_config=generation_config)
    print("✅ Gemini model loaded")
except Exception as e:
    print(f"❌ Error loading models: {str(e)}")
    raise

# Sample insurance policy chunks
chunks = [
    "Deductible refers to the fixed amount that the insured person must pay before the insurance company starts covering the expenses. Co-payment is a percentage of the claim amount that the insured person must pay.",
    "Waiting period: A waiting period is a time period in which you cannot make claims. For example, most health insurance policies have a 30-day waiting period for illnesses.",
    "Pre-existing diseases: Conditions, ailments, or injuries that existed before the policy was purchased. These may be covered after a waiting period.",
    "Premium: The amount paid by the policyholder to the insurance company for providing insurance coverage. It can be paid monthly, quarterly, or annually.",
    "Coverage limit: The maximum amount an insurance company will pay toward a covered loss. Once this limit is reached, the policyholder must pay all additional costs."
]

# Pre-compute embeddings once at startup
chunk_embeddings = embedding_model.encode(chunks)

@app.route('/api/query', methods=['POST'])
def query():
    try:
        # Get query from request
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
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:3]
        matches = []
        for idx in top_indices:
            matches.append({
                "content": chunks[idx],
                "score": float(similarities[idx]),
                "source": f"Policy_Document_{idx+1}.txt"
            })
        
        # Generate AI response
        prompt = f"""
        Query: {query}
        
        Relevant policy information:
        {' '.join([m['content'] for m in matches])}
        
        Please provide a response in this exact JSON format:
        {{
            "decision": "Approved/Rejected/Need more info",
            "justification": "Brief explanation based on policy",
            "amount": "Amount if applicable, otherwise N/A",
            "next_steps": "Recommended actions"
        }}
        """
        
        # Generate response with retry logic and API key rotation
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Configure with next API key on retry
                api_key = get_next_api_key()
                genai.configure(api_key=api_key)
                print(f"🔄 Attempt {attempt + 1} with new API key")
                
                response = gemini_model.generate_content(prompt)
                ai_response = response.text.strip()
                
                # Try to parse AI response as JSON
                try:
                    ai_response = json.loads(ai_response)
                    print("✅ Successfully generated response")
                    break  # If successful, exit the retry loop
                except json.JSONDecodeError:
                    print(f"⚠️ Failed to parse JSON response on attempt {attempt + 1}")
                    if attempt == max_retries - 1:  # If last attempt
                        print(f"❌ Failed to parse JSON response after {max_retries} attempts")
                        ai_response = {
                            "decision": "Need more info",
                            "justification": "Unable to process model response",
                            "amount": "N/A",
                            "next_steps": "Please try again or contact support"
                        }
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:  # If last attempt
                    raise
        
        return jsonify({
            "matches": matches,
            "ai_response": ai_response
        })
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)