import faiss
import pickle
import re
from rank_bm25 import BM25Okapi
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, '..', 'models')


def tokenize(text):
    """Lowercase alphanumeric tokenization for BM25.
    MUST stay identical to tokenize() in app/backend/app.py so the query and the
    corpus are tokenized the same way."""
    return re.findall(r"[a-z0-9]+", text.lower())


# Load embeddings and chunks
embeddings = pickle.load(open(os.path.join(models_dir, 'embeddings.pkl'), 'rb'))
chunks = pickle.load(open(os.path.join(models_dir, 'chunks.pkl'), 'rb'))

# Create BM25 index for re-ranking (chunks are {"text", "page"} dicts)
tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)
pickle.dump(bm25, open(os.path.join(models_dir, 'bm25.pkl'), 'wb'))

# Normalize embeddings for cosine similarity
faiss.normalize_L2(embeddings)

d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)  # Inner Product ~ Cosine similarity
index.add(embeddings)

faiss.write_index(index, os.path.join(models_dir, 'faiss_index.index'))
print("FAISS index and BM25 created successfully!")
