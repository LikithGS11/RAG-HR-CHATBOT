import faiss
import pickle
import numpy as np
from rank_bm25 import BM25Okapi

# Load embeddings and chunks
embeddings = pickle.load(open('models/embeddings.pkl', 'rb'))
chunks = pickle.load(open('models/chunks.pkl', 'rb'))

# Create BM25 index for re-ranking
tokenized_chunks = [chunk.split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)
pickle.dump(bm25, open('models/bm25.pkl', 'wb'))

# Normalize embeddings for cosine similarity
faiss.normalize_L2(embeddings)

d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)  # Inner Product ~ Cosine similarity
index.add(embeddings)

faiss.write_index(index, 'models/faiss_index.index')
print("FAISS index and BM25 created successfully!")
