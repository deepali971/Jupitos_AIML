#2. Embeddings
#Word2Vec Implementation
from gensim.models import Word2Vec

sentences = [["king", "queen", "man", "woman", "royalty"],
             ["apple", "banana", "orange", "fruit"],
             ["dog", "cat", "pet"]]

# Train Word2Vec
model = Word2Vec(sentences, vector_size=50, window=5, min_count=1, workers=4)

# Get vector for 'king'
vector = model.wv["king"]

print("Word2Vec Embedding for 'king':")
print(vector)
