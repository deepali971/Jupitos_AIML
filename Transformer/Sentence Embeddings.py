#5.Sentence Embeddings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = ["This is an example sentence.", "This is another one."]
embeddings = model.encode(sentences)

print("Sentence Embeddings:")
for i, emb in enumerate(embeddings):
    print(f"Sentence {i+1}:", emb)
