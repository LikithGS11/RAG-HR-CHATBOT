from flask import Flask, request, jsonify
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
import re
import logging
from dotenv import load_dotenv
import uuid
import time
from threading import Lock

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("anchor")

app = Flask(__name__)


def tokenize(text):
    """Lowercase alphanumeric tokenization for BM25.
    MUST stay identical to tokenize() in utils/faiss_index.py so the query and
    the corpus are tokenized the same way."""
    return re.findall(r"[a-z0-9]+", text.lower())

# --- Configuration -----------------------------------------------------------
MAX_QUERY_LENGTH = 2000            # reject oversized queries
MAX_HISTORY_MESSAGES = 20          # cap conversation turns sent to the LLM
CACHE_EXPIRATION_SECONDS = 300     # 5 minutes
SESSION_EXPIRATION_SECONDS = 3600  # evict idle sessions after 1 hour
RATE_LIMIT_MAX_REQUESTS = 30       # per session, per window
RATE_LIMIT_WINDOW_SECONDS = 60
# Optional shared secret. If set, callers must send it as the X-API-Key header.
API_KEY = os.getenv('APP_API_KEY')

# Load models
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, '../../models')
embeddings = pickle.load(open(os.path.join(models_dir, 'embeddings.pkl'), 'rb'))
chunks = pickle.load(open(os.path.join(models_dir, 'chunks.pkl'), 'rb'))
bm25 = pickle.load(open(os.path.join(models_dir, 'bm25.pkl'), 'rb'))
index = faiss.read_index(os.path.join(models_dir, 'faiss_index.index'))
model = SentenceTransformer('all-MiniLM-L6-v2')
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# In-memory session chat histories: {session_id: {"messages": [...], "last_seen": ts}}
chat_histories = {}
chat_histories_lock = Lock()

# Simple in-memory cache: {(session_id, history_len, query): {"answer", "sources", "timestamp"}}
cache = {}
cache_lock = Lock()

# Per-session request timestamps for rate limiting: {session_id: [ts, ...]}
rate_limits = {}
rate_limits_lock = Lock()


def cleanup_cache():
    now = time.time()
    with cache_lock:
        keys_to_delete = [k for k, v in cache.items() if now - v["timestamp"] > CACHE_EXPIRATION_SECONDS]
        for k in keys_to_delete:
            del cache[k]


def cleanup_sessions():
    """Evict idle sessions so memory does not grow without bound."""
    now = time.time()
    with chat_histories_lock:
        stale = [sid for sid, s in chat_histories.items()
                 if now - s["last_seen"] > SESSION_EXPIRATION_SECONDS]
        for sid in stale:
            del chat_histories[sid]
    with rate_limits_lock:
        for sid in list(rate_limits.keys()):
            rate_limits[sid] = [t for t in rate_limits[sid] if now - t < RATE_LIMIT_WINDOW_SECONDS]
            if not rate_limits[sid]:
                del rate_limits[sid]


def rate_limited(session_id):
    """Return True if this session has exceeded its request budget."""
    now = time.time()
    with rate_limits_lock:
        window = [t for t in rate_limits.get(session_id, []) if now - t < RATE_LIMIT_WINDOW_SECONDS]
        if len(window) >= RATE_LIMIT_MAX_REQUESTS:
            rate_limits[session_id] = window
            return True
        window.append(now)
        rate_limits[session_id] = window
        return False


def _min_max_normalize(scores):
    """Scale an array of scores into [0, 1]; flat arrays map to 0.5."""
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if hi - lo < 1e-9:
        return np.full_like(scores, 0.5, dtype=np.float32)
    return (scores - lo) / (hi - lo)


def retrieve(query, top_k=5, candidate_pool=20):
    # Encode query
    query_emb = model.encode([query]).astype(np.float32)
    faiss.normalize_L2(query_emb)

    # FAISS search over a wider candidate pool than we ultimately return
    pool = min(candidate_pool, index.ntotal)
    distances, indices = index.search(query_emb, pool)

    # BM25 scores across the whole corpus (tokenized to match the index)
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    # Union of candidates: top FAISS hits AND top BM25 hits, so neither
    # retriever can hide a document the other would have surfaced.
    faiss_ids = [int(i) for i in indices[0] if i != -1]
    bm25_top = [int(i) for i in np.argsort(bm25_scores)[::-1][:pool]]
    candidate_ids = list(dict.fromkeys(faiss_ids + bm25_top))

    # Normalize both score sets to a common [0, 1] scale before combining.
    faiss_by_id = {int(i): float(d) for i, d in zip(indices[0], distances[0]) if i != -1}
    cand_faiss = np.array([faiss_by_id.get(i, 0.0) for i in candidate_ids], dtype=np.float32)
    cand_bm25 = np.array([bm25_scores[i] for i in candidate_ids], dtype=np.float32)
    norm_faiss = _min_max_normalize(cand_faiss)
    norm_bm25 = _min_max_normalize(cand_bm25)
    combined = norm_faiss + norm_bm25

    order = np.argsort(combined)[::-1][:top_k]
    retrieved_chunks = [{"id": candidate_ids[j], "text": chunks[candidate_ids[j]]} for j in order]
    return retrieved_chunks

SYSTEM_INSTRUCTION = (
    "You are Anchor, a friendly and professional HR assistant.\n"
    "For greetings, thanks, or small talk (e.g. 'hi', 'thanks'), respond warmly "
    "and briefly and invite an HR question — do NOT mention documents or context.\n"
    "For HR questions:\n"
    "- Base every fact ONLY on the provided context. Never invent numbers, names, "
    "policies, or details that are not in it.\n"
    "- Do NOT copy the source text word-for-word. Rewrite it in your own natural, "
    "conversational voice, the way a helpful HR colleague would explain it to an "
    "employee.\n"
    "- Structure the answer well: a short direct opener, then bullet points for "
    "multiple rules or figures, and a brief friendly closing offer to help further.\n"
    "- You may explain and add helpful framing for clarity, but the underlying facts "
    "must come only from the context.\n"
    "- If the context doesn't contain the answer, say so plainly and suggest how to "
    "rephrase the question."
)


def _authorized():
    """When API_KEY is configured, require a matching X-API-Key header."""
    if not API_KEY:
        return True
    return request.headers.get("X-API-Key") == API_KEY


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "chunks": index.ntotal})


@app.route('/query', methods=['POST'])
def query():
    if not _authorized():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id") or str(uuid.uuid4())
    query = data.get('query')
    if not query or not isinstance(query, str) or not query.strip():
        return jsonify({"error": "Missing 'query' parameter"}), 400
    query = query.strip()
    if len(query) > MAX_QUERY_LENGTH:
        return jsonify({"error": f"Query exceeds {MAX_QUERY_LENGTH} characters"}), 413

    if rate_limited(session_id):
        return jsonify({"error": "Rate limit exceeded, slow down"}), 429

    # Housekeeping
    cleanup_cache()
    cleanup_sessions()

    # Snapshot prior history so the cache key reflects conversational state.
    with chat_histories_lock:
        session = chat_histories.get(session_id)
        prior = list(session["messages"]) if session else []
    cache_key = (session_id, len(prior), query)

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

    # Build messages: system instruction first, prior turns, then the
    # context-grounded question as the current user turn.
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    messages.extend(prior[-MAX_HISTORY_MESSAGES:])
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}"
    })

    # Call LLM
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.6,
            max_tokens=800,
        )
        answer = response.choices[0].message.content
    except Exception:
        logger.exception("LLM generation failed for session %s", session_id)
        return jsonify({"error": "The assistant is temporarily unavailable. Please try again."}), 502

    # Persist the turn
    with chat_histories_lock:
        session = chat_histories.setdefault(session_id, {"messages": [], "last_seen": time.time()})
        session["messages"].append({"role": "user", "content": query})
        session["messages"].append({"role": "assistant", "content": answer})
        session["last_seen"] = time.time()

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
    if not _authorized():
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    if session_id:
        with chat_histories_lock:
            chat_histories.pop(session_id, None)
        with cache_lock:
            keys_to_delete = [k for k in cache if k[0] == session_id]
            for k in keys_to_delete:
                del cache[k]
        with rate_limits_lock:
            rate_limits.pop(session_id, None)
    return jsonify({"status": "reset done"})


if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)
