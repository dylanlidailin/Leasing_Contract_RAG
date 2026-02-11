import os

# Fix OpenMP crashes on Mac
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import asyncio
import json
import uuid
import time
import hashlib
import pickle
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

# Unstructured Imports
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY: raise ValueError("Missing API Key")
genai.configure(api_key=API_KEY)

# Ensure cache directory exists
CACHE_DIR = Path("document_cache")
CACHE_DIR.mkdir(exist_ok=True)

# ==========================================
# 1. CACHING UTILITIES
# ==========================================
def get_file_hash(filepath: str) -> str:
    """Generates a unique MD5 hash for the file content."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def save_to_cache(file_hash, data):
    """Saves parsed chunks and vectors to disk."""
    cache_path = CACHE_DIR / f"{file_hash}.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[+] Saved to cache: {cache_path}")

def load_from_cache(file_hash):
    """Attempts to load data from disk."""
    cache_path = CACHE_DIR / f"{file_hash}.pkl"
    if cache_path.exists():
        print(f"[+] Cache Hit! Loading from {cache_path}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None

# ==========================================
# 2. INGESTION (With Cache Check)
# ==========================================
def process_document(pdf_path: str):
    # 1. Calculate Hash
    file_hash = get_file_hash(pdf_path)
    
    # 2. Check Cache
    cached_data = load_from_cache(file_hash)
    if cached_data:
        return cached_data['chunks'], cached_data['vectors']

    # 3. Cache Miss - Run Heavy Processing
    print(f"[+] Cache Miss. Parsing '{pdf_path}' locally...")
    
    # A. Parse (Expensive CPU)
    elements = partition_pdf(
        filename=pdf_path,
        strategy="fast",
        infer_table_structure=False,
        include_page_breaks=False
    )
    
    chunks_obj = chunk_by_title(
        elements,
        max_characters=1000,
        new_after_n_chars=800,
        combine_text_under_n_chars=500
    )
    text_chunks = [chunk.text for chunk in chunks_obj]
    print(f"    Created {len(text_chunks)} chunks.")

    # B. Embed (Expensive API Cost)
    print("    Generating embeddings (API Call)...")
    result = genai.embed_content(
        model="models/embedding-001",
        content=text_chunks,
        task_type="retrieval_document"
    )
    vectors = np.array(result['embedding'])

    # 4. Save to Cache
    save_to_cache(file_hash, {'chunks': text_chunks, 'vectors': vectors})
    
    return text_chunks, vectors

# ==========================================
# 3. SEARCH & GENERATION (Standard)
# ==========================================
def get_query_embedding(text):
    result = genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_query")
    return np.array(result['embedding'])

def find_best_chunks(query_vec, doc_vecs, chunks, k=5):
    dot_product = np.dot(doc_vecs, query_vec)
    top_indices = np.argsort(dot_product)[::-1][:k]
    return [chunks[i] for i in top_indices]

def generate_answer(context, question):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    try:
        return model.generate_content(prompt).text.strip()
    except Exception:
        return "Error generating response."

# ==========================================
# MAIN APP
# ==========================================
async def main():
    print("========================================")
    print("  Local RAG + Smart Caching")
    print("========================================")
    
    while True:
        pdf_path = input("\nEnter PDF path: ").strip()
        if os.path.exists(pdf_path): break
        print("File not found.")

    # This single line handles checking cache vs. processing new
    chunks, doc_vectors = process_document(pdf_path)

    print("\n[+] Ready for Chat!")
    while True:
        q = input("You: ").strip()
        if q.lower() == 'exit': break
        
        q_vec = get_query_embedding(q)
        retrieved = find_best_chunks(q_vec, doc_vectors, chunks)
        ans = generate_answer("\n".join(retrieved), q)
        print(f"Bot: {ans}")

if __name__ == "__main__":
    asyncio.run(main())