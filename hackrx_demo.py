from sentence_transformers import SentenceTransformer
import numpy as np

# Sample insurance policy chunks (simulated)
chunks = [
    "Deductible refers to the fixed amount that the insured person must pay before the insurance company starts covering the expenses. Co-payment is a percentage of the claim amount that the insured person must pay, with the insurance company paying the rest.",
    "Waiting period: A waiting period is a time period in which you cannot make claims. For example, most health insurance policies have a 30-day waiting period for illnesses and 24-48 months for pre-existing diseases.",
    "Pre-existing diseases: Conditions, ailments, or injuries that existed before the policy was purchased. These may be covered after a waiting period of 24-48 months depending on the insurance provider.",
    "Premium: The amount paid by the policyholder to the insurance company for providing insurance coverage. It can be paid monthly, quarterly, or annually.",
    "Coverage limit: The maximum amount an insurance company will pay toward a covered loss. Once this limit is reached, the policyholder must pay all additional costs.",
]

# Metadata for chunks
metadata = [
    {"source": "policy_definitions.txt"},
    {"source": "waiting_periods.txt"},
    {"source": "exclusions.txt"},
    {"source": "premium_info.txt"},
    {"source": "coverage_details.txt"},
]

print("🔄 Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed chunks (simulating database)
print("🔄 Embedding policy chunks...")
chunk_embeddings = model.encode(chunks)

# User query
query = "What is deductible or co-payment?"
print(f"❓ Query: {query}")

# Embed query
query_embedding = model.encode(query)
print("✅ Query embedded successfully")

# Calculate similarity scores
print("🔄 Calculating semantic similarity...")
similarities = []
for chunk_embedding in chunk_embeddings:
    # Cosine similarity
    similarity = np.dot(query_embedding, chunk_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding))
    similarities.append(similarity)

# Get top 3 most similar chunks
top_indices = np.argsort(similarities)[::-1][:3]
top_chunks = [chunks[i] for i in top_indices]
top_metadata = [metadata[i] for i in top_indices]
top_scores = [similarities[i] for i in top_indices]

print("\n🔍 Top Matching Chunks:\n")
for i, (chunk, meta, score) in enumerate(zip(top_chunks, top_metadata, top_scores)):
    print(f"Match #{i+1} (Score: {score:.4f})")
    print(f"📄 Source: {meta['source']}")
    print(f"📝 Content:\n{chunk}\n")

print("\n✅ Demo complete!")

# For the next phase of HackRx
print("\n💡 Next steps for HackRx hackathon:")
print("1. Integrate with Gemini API for insights")
print("2. Create interactive UI for queries")
print("3. Prepare final pitch deck")