import chromadb
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client & collection
client = chromadb.PersistentClient(path="./chromadb_store")

collection = client.get_or_create_collection(name="hackrx_docs")


# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Ask user query
  # query = input("🤖 Enter your question: ")
query = "What is deductible or co-payment?"
# Embed query
query_embedding = model.encode(query)

# Perform semantic search
try:
    print('flag2')
    # First try the query
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        print('flag3')
        print("DEBUG: Query ran successfully")
    except Exception as e:
        import traceback
        print("❌ Detailed Error during query:")
        traceback.print_exc()
        raise e
except Exception as e:
    print("❌ Error during query:", e)
    # Create dummy results for demonstration
    results = {
        'ids': [['dummy_id_1', 'dummy_id_2', 'dummy_id_3']],
        'documents': [['Deductible refers to the fixed amount that the insured person must pay before the insurance company starts covering the expenses. Co-payment is a percentage of the claim amount that the insured person must pay, with the insurance company paying the rest.', 'Additional information about deductibles...', 'Further details about co-payments...']], 
        'metadatas': [[{'source': 'doc1.txt_chunk10.txt'}, {'source': 'doc2.txt_chunk5.txt'}, {'source': 'doc3.txt_chunk8.txt'}]],
        'distances': [[0.1, 0.2, 0.3]]
    }
    print("Using dummy results for demonstration")

print("DEBUG: Total docs in collection:", collection.count())
print("DEBUG: Query results keys:", results.keys())
print("DEBUG: Documents:", results['documents'])
print("DEBUG: Metadatas:", results['metadatas'])

# Show response
print("\n🔍 Top Matching Chunks:\n")
if not results['documents'][0]:
    print("❌ No matching documents found.")
else:
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        print(f"📄 Source: {meta['source']}\n📝 Content:\n{doc}\n")
