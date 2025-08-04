import chromadb
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client & collection
print("🔄 Initializing ChromaDB client...")
client = chromadb.PersistentClient(path="./chromadb_store")

print("🔄 Getting collection...")
collection = client.get_or_create_collection(name="hackrx_docs")

print("📊 Collection count:", collection.count())

# Load embedding model
print("🔄 Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Ask user query
query = "What is deductible or co-payment?"
print(f"❓ Query: {query}")

# Embed query
query_embedding = model.encode(query)
print("✅ Query embedded successfully")

# Skip query and use dummy results
print("⚠️ Using dummy results for demonstration")
results = {
    'ids': [['dummy_id_1', 'dummy_id_2', 'dummy_id_3']],
    'documents': [['Deductible refers to the fixed amount that the insured person must pay before the insurance company starts covering the expenses. Co-payment is a percentage of the claim amount that the insured person must pay, with the insurance company paying the rest.', 'Additional information about deductibles...', 'Further details about co-payments...']], 
    'metadatas': [[{'source': 'doc1.txt_chunk10.txt'}, {'source': 'doc2.txt_chunk5.txt'}, {'source': 'doc3.txt_chunk8.txt'}]],
    'distances': [[0.1, 0.2, 0.3]]
}

print("\n🔍 Top Matching Chunks:\n")
if not results['documents'][0]:
    print("❌ No matching documents found.")
else:
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        print(f"📄 Source: {meta['source']}\n📝 Content:\n{doc}\n")

print("\n✅ Demo complete!")