# embeddings_utils.py
import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np, tiktoken

__all__ = ["embed_article", "chunk_by_tokens", "embed_texts"]

load_dotenv()  # Optional .env support
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def _norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)

def chunk_by_tokens(text: str, model: str, chunk_size: int = 800, overlap: int = 400):
    enc = tiktoken.encoding_for_model(model)
    toks = enc.encode(text or "")
    step = max(1, chunk_size - overlap)
    return [enc.decode(toks[i:i+chunk_size]) for i in range(0, len(toks), step)]

def embed_texts(texts, model="text-embedding-3-small", dtype=np.float32):
    if not texts:
        return np.zeros((0, 1536), dtype=dtype)
    resp = client.embeddings.create(model=model, input=texts)
    return np.asarray([d.embedding for d in resp.data], dtype=dtype)

def embed_article(title: str, full_text: str,
                  model: str = "text-embedding-3-small",
                  chunk_size: int = 800, overlap: int = 400,
                  top_k: int = 3):
    title = (title or "").strip()
    body = (full_text or "").strip()

    enc = tiktoken.encoding_for_model(model)
    lede = enc.decode(enc.encode(body)[:200])
    query = (title + " — " + lede).strip() or body[:400]

    chunks = chunk_by_tokens(body, model, chunk_size, overlap)

    # Fallback: no chunks → query-only embedding
    if not chunks:
        v = embed_texts([query] if query else [body], model=model)
        return {
            "doc_vector": _norm(v)[0],
            "chunk_vectors": np.zeros((0, v.shape[1]), dtype=v.dtype),
            "chosen_indices": [],
            "chosen_scores": np.array([], dtype=np.float32),
            "chunks": [],
            "used_fallback": True,
            "fallback_reason": "no_chunks",
            "method": "query_only",
        }

    embs = embed_texts([query] + chunks, model=model)
    q = _norm(embs[:1])[0]
    C = _norm(embs[1:])
    scores = C @ q

    k = max(1, min(top_k, C.shape[0]))
    idx = np.argpartition(-scores, k-1)[:k]
    idx = idx[np.argsort(-scores[idx])]

    w = np.clip(scores[idx], 0.0, None)
    if w.sum() == 0:  # Fallback: all similarities non-positive → uniform weights
        w = np.ones_like(w)
        fb_used, fb_reason = True, "non_positive_similarity"
    else:
        fb_used, fb_reason = False, None

    doc_vec = _norm((C[idx] * w[:, None]).sum(axis=0, keepdims=True))[0]

    return {
        "doc_vector": doc_vec,
        "chunk_vectors": C,
        "chosen_indices": idx.tolist(),
        "chosen_scores": scores[idx],
        "chunks": [chunks[i] for i in idx],
        "used_fallback": fb_used,
        "fallback_reason": fb_reason,
        "method": "hybrid",
    }
