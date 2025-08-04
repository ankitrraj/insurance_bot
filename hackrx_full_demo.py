import os
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Set API key directly
API_KEY = "AIzaSyAhxI1iwIiOdO2nlJF4q0dleRbELx0-CGY"

print("📂 Loading all chunks from file system...")
chunks = []
metadata = []
chunks_dir = "./chunks"

# Load all chunks from the chunks directory
for filename in os.listdir(chunks_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(chunks_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:  # Only add non-empty chunks
                chunks.append(content)
                metadata.append({"source": filename})

print(f"✅ Loaded {len(chunks)} chunks from file system")

print("🔄 Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed chunks
print(f"🔄 Embedding {len(chunks)} chunks... (this may take a moment)")
chunk_embeddings = model.encode(chunks)
print("✅ All chunks embedded successfully")

# User query
default_query = "What is deductible or co-payment?"
user_query = input(f"🤖 Enter your question (or press Enter for default '{default_query}'): ")
query = user_query if user_query.strip() else default_query
print(f"❓ Query: {query}")

# Embed query
query_embedding = model.encode(query)
print("✅ Query embedded successfully")

# Calculate similarity scores
print("🔄 Calculating semantic similarity across all chunks...")
similarities = []
for chunk_embedding in chunk_embeddings:
    # Cosine similarity
    similarity = np.dot(query_embedding, chunk_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding))
    similarities.append(similarity)

# Get top 5 most similar chunks
top_k = 5
top_indices = np.argsort(similarities)[::-1][:top_k]
top_chunks = [chunks[i] for i in top_indices]
top_metadata = [metadata[i] for i in top_indices]
top_scores = [similarities[i] for i in top_indices]

print(f"\n🔍 Top {top_k} Matching Chunks from {len(chunks)} total chunks:\n")
for i, (chunk, meta, score) in enumerate(zip(top_chunks, top_metadata, top_scores)):
    print(f"Match #{i+1} (Score: {score:.4f})")
    print(f"📄 Source: {meta['source']}")
    print(f"📝 Content (first 200 chars):\n{chunk[:200]}...\n")

# Use Gemini API to get response
print("\n🧠 Generating AI-powered answer with Gemini...\n")

try:
    import google.generativeai as genai
    genai.configure(api_key=API_KEY)
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Prepare content for the prompt (combine top chunks)
    top_chunks_content = ""
    for i, (chunk, meta) in enumerate(zip(top_chunks, top_metadata)):
        top_chunks_content += f"Clause {i+1} from {meta['source']}:\n{chunk}\n\n"
    
    prompt = f"""
    Query from user: {query}
    
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
    
    response = model.generate_content(prompt)
    print(response.text)
except Exception as e:
    print(f"⚠️ Error with Gemini API: {e}")
    print("\nFalling back to simulated response:")
    
    simulated_response = json.dumps({
        "decision": "Need more info",
        "justification": "Based on the top matching policy clauses, additional information is needed about your specific situation to determine eligibility.",
        "amount": "N/A (depends on specific policy terms)",
        "next_steps": "Please contact your insurance provider with your policy number and details of your situation."
    }, indent=2)
    print(simulated_response)

print("\n✅ Demo complete!")
print("\n💡 HackRx submission is ready!")