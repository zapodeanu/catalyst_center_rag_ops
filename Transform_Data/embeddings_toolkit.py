from sentence_transformers import SentenceTransformer

sentences = ["interface Loopback0\n ip address 10.93.141.20 255.255.255.255"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings)
