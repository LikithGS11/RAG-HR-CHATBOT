from flask import Flask, request, jsonify
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from groq import Groq
import os
from dotenv import load_dotenv
import uuid
import time
from threading import Lock

load_dotenv()

app = Flask(__name__)

# Load models
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, '../../models')
embeddings = pickle.load(open(os.path.join(models_dir, 'embeddings.pkl'), 'rb'))
chunks = pickle.load(open(os.path.join(models_dir, 'chunks.pkl'), 'rb'))
bm25 = pickle.load(open(os.path.join(models_dir, 'bm25.pkl'), 'rb'))
index = faiss.read_index(os.path.join(models_dir, 'faiss_index.index'))
model = SentenceTransformer('all-MiniLM-L6-v2')
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# In-memory session chat histories: {session_id: [{"role": "user"/"assistant", "content": "..."}]}
chat_histories = {}
chat_histories_lock = Lock()

# Simple in-memory cache: {(session_id, query): {"answer": ..., "sources": ..., "timestamp": ...}}
cache = {}
cache_lock = Lock()
CACHE_EXPIRATION_SECONDS = 300  # 5 minutes

def cleanup_cache():
    now = time.time()
    with cache_lock:
        keys_to_delete = [k for k, v in cache.items() if now - v["timestamp"] > CACHE_EXPIRATION_SECONDS]
        for k in keys_to_delete:
            del cache[k]

def retrieve(query, top_k=5):
    # Encode query
    query_emb = model.encode([query])
    faiss.normalize_L2(query_emb)
    
    # FAISS search
    distances, indices = index.search(query_emb, top_k)
    
    # BM25 scores
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Hybrid scoring
    combined_scores = {}
    for i, dist in zip(indices[0], distances[0]):
        combined_scores[i] = dist + bm25_scores[i]
    
    # Sort by combined score
    sorted_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)[:top_k]
    # Convert numpy int64 to int for JSON serialization
    retrieved_chunks = [{"id": int(i), "text": chunks[i]} for i in sorted_indices]
    
    return retrieved_chunks

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    session_id = data.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
    query = data['query']
    
    # Check cache
    cleanup_cache()
    cache_key = (session_id, query)
    with cache_lock:
        if cache_key in cache:
            cached_response = cache[cache_key]
            return jsonify({
                "answer": cached_response["answer"],
                "sources": cached_response["sources"],
                "session_id": session_id
            })
    
    # Retrieve context chunks with IDs
    retrieved = retrieve(query)
    context = '\n'.join([chunk["text"] for chunk in retrieved])
    
    # Manage chat history
    with chat_histories_lock:
        history = chat_histories.get(session_id, [])
        history.append({"role": "user", "content": query})
        # Prepare messages for LLM: include previous conversation + current prompt
        messages = []
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        # Add context and current question as system prompt
        system_prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        messages.append({"role": "system", "content": system_prompt})
    
    # Call LLM
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )
    answer = response.choices[0].message.content
    
    # Update chat history with assistant response
    with chat_histories_lock:
        history.append({"role": "assistant", "content": answer})
        chat_histories[session_id] = history
    
    # Cache the response
    with cache_lock:
        cache[cache_key] = {
            "answer": answer,
            "sources": retrieved,
            "timestamp": time.time()
        }
    
    return jsonify({
        "answer": answer,
        "sources": retrieved,
        "session_id": session_id
    })

@app.route('/reset', methods=['POST'])
def reset():
    data = request.json
    session_id = data.get("session_id")
    if session_id:
        with chat_histories_lock:
            if session_id in chat_histories:
                del chat_histories[session_id]
        with cache_lock:
            keys_to_delete = [k for k in cache if k[0] == session_id]
            for k in keys_to_delete:
                del cache[k]
    return jsonify({"status": "reset done"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
