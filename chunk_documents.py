import os

# Input folder
folder_path = "./data"
output_folder = "./chunks"
os.makedirs(output_folder, exist_ok=True)

# Function to split text into chunks (paragraph-wise)
def split_into_chunks(text, chunk_size=1000):
    paragraphs = text.split('\n\n')
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current += para + "\n\n"
        else:
            chunks.append(current.strip())
            current = para + "\n\n"
    if current:
        chunks.append(current.strip())
    return chunks

# Process all text files
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
            text = f.read()
            chunks = split_into_chunks(text)
        
        for i, chunk in enumerate(chunks):
            out_path = os.path.join(output_folder, f"{filename}_chunk{i+1}.txt")
            with open(out_path, "w", encoding="utf-8") as out_f:
                out_f.write(chunk)

print("✅ Chunking complete. Chunks saved in 'chunks/' folder.")