import json
from sentence_transformers import SentenceTransformer
import pickle

def create_embeddings(chunks_file, model_name='all-MiniLM-L6-v2'):
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    with open('models/embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    with open('models/chunks.pkl', 'wb') as f:
        pickle.dump(chunks, f)

if __name__ == "__main__":
    create_embeddings('hr_chunks.json')
