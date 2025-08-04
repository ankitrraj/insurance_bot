import os
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

# Step 1: Load Chunks
chunks_dir = "./chunks"
documents = []
metadata = []

for filename in os.listdir(chunks_dir):
    if filename.endswith(".txt"):
        path = os.path.join(chunks_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:  # Only add non-empty chunks
                documents.append(content)
                metadata.append({"source": filename})

print(f"📦 Total chunks loaded: {len(documents)}")

# Step 2: Remove empty/whitespace-only chunks (extra safety)
filtered = [
    (doc, meta) for doc, meta in zip(documents, metadata)
    if doc and doc.strip()
]
if not filtered:
    print("❌ No non-empty chunks found! Exiting.")
    exit()
documents, metadata = zip(*filtered)
documents = list(documents)
metadata = list(metadata)
print(f"✅ Non-empty chunks after filtering: {len(documents)}")

# Step 3: Generate Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)
print(f"🔢 Embeddings generated for all non-empty chunks.")

# Step 4: Convert embeddings to list (for ChromaDB compatibility)
if isinstance(embeddings, np.ndarray):
    embeddings = embeddings.tolist()
print(f"Sample embedding shape: {len(embeddings[0])} (should be 384 for MiniLM-L6-v2)")
print(f"Type of embeddings[0]: {type(embeddings[0])}")
print(f"Type of embeddings: {type(embeddings)}")

# Step 5: Store in ChromaDB (Persistent, batch insert)
client = chromadb.PersistentClient(path="./chromadb_store")
collection = client.get_or_create_collection(name="hackrx_docs")

try:
    batch_size = 16
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_embeds = embeddings[i:i+batch_size]
        batch_ids = [f"doc_{j}" for j in range(i, i+len(batch_docs))]
        batch_metas = metadata[i:i+batch_size]
        print(f"Inserting batch {i//batch_size + 1}: {len(batch_docs)} docs")
        try:
            collection.add(
                documents=batch_docs,
                embeddings=batch_embeds,
                ids=batch_ids,
                metadatas=batch_metas
            )
            print(f"✅ Inserted batch {i//batch_size + 1} ({len(batch_docs)} docs)")
        except Exception as e:
            print(f"❌ Error inserting batch {i//batch_size + 1}: {e}")
    print(f"✅ All embeddings inserted into collection '{collection.name}'")
    print(f"🔍 Total docs in collection: {collection.count()}")
except Exception as e:
    print("❌ Error inserting into ChromaDB:", e)
