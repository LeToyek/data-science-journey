from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = ["I love dogs", "I love cats", "I love programming"]

embeddings = model.encode(sentences)

print(embeddings)
