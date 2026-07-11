import json
from sentence_transformers import SentenceTransformer
import pickle
import os

def create_embeddings(chunks_file, models_dir, model_name='all-MiniLM-L6-v2'):
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'embeddings.pkl'), 'wb') as f:
        pickle.dump(embeddings, f)
    with open(os.path.join(models_dir, 'chunks.pkl'), 'wb') as f:
        pickle.dump(chunks, f)
    print("Embeddings and chunks saved.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data')
    models_dir = os.path.join(base_dir, '..', 'models')
    
    chunks_file = os.path.join(data_dir, 'hr_chunks.json')
    create_embeddings(chunks_file, models_dir)
