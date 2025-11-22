import os

# 1. Fix OpenMP crashes on Mac (Must be before other imports)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import asyncio
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

# Unstructured Imports
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env")

genai.configure(api_key=API_KEY)

# ==========================================
# 1. LOCAL INGESTION (Unstructured)
# ==========================================
def load_and_chunk_locally(pdf_path: str):
    print(f"\n[+] Parsing '{pdf_path}' locally...")
    
    # Use "fast" strategy to avoid Tesseract crashes if "hi_res" is unstable
    elements = partition_pdf(
        filename=pdf_path,
        strategy="fast",             
        infer_table_structure=False,  
        include_page_breaks=False
    )
    
    print(f"    Found {len(elements)} raw elements.")

    chunks = chunk_by_title(
        elements,
        max_characters=1000,
        new_after_n_chars=800,
        combine_text_under_n_chars=500
    )
    
    text_chunks = [chunk.text for chunk in chunks]
    print(f"[+] Created {len(text_chunks)} chunks.")
    return text_chunks

# ==========================================
# 2. NATIVE EMBEDDINGS
# ==========================================
def get_embeddings(texts):
    # Embed documents
    result = genai.embed_content(
        model="models/embedding-001",
        content=texts,
        task_type="retrieval_document"
    )
    return np.array(result['embedding'])

def get_query_embedding(text):
    # Embed query
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_query"
    )
    return np.array(result['embedding'])

# ==========================================
# 3. SIMPLE VECTOR SEARCH (No FAISS)
# ==========================================
def find_best_chunks(query_vec, doc_vecs, chunks, k=5):
    """
    Performs simple Cosine Similarity using Numpy.
    Crash-proof replacement for FAISS.
    """
    # Calculate Dot Product
    dot_product = np.dot(doc_vecs, query_vec)
    
    # Get indices of top k scores (sorted descending)
    top_indices = np.argsort(dot_product)[::-1][:k]
    
    return [chunks[i] for i in top_indices]

# ==========================================
# 4. GENERATION
# ==========================================
def generate_answer(context, question):
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    You are an expert lease analyst.
    
    RULES:
    1. Use the context provided below.
    2. If the context shows a label like "Tenant:" but no name, state that it is blank.
    3. 'Buyer/Tenant' refers to the Tenant.
    
    Context:
    {context}
    
    Question: {question}
    Answer:
    """
    
    response = model.generate_content(prompt)
    return response.text

# ==========================================
# MAIN APP
# ==========================================
async def main():
    print("========================================")
    print("  Local RAG (Numpy Version - Crash Proof)")
    print("========================================")
    
    # 1. Input
    while True:
        pdf_path = input("\nEnter PDF path: ").strip()
        if os.path.exists(pdf_path) and pdf_path.lower().endswith('.pdf'):
            break
        print("[!] File not found or not a PDF.")

    # 2. Ingest
    try:
        chunks = load_and_chunk_locally(pdf_path)
    except Exception as e:
        print(f"[!] Parsing Error: {e}")
        return

    # 3. Embed
    print("[+] Embedding chunks...")
    # Vectors are just a standard numpy array now
    doc_vectors = get_embeddings(chunks)

    # 4. Chat Loop
    print("\n[+] Ready! Type 'exit' to quit.")
    
    while True:
        try:
            question = input("\nYou: ")
            if question.lower() in ['exit', 'quit']: break
            if not question.strip(): continue

            # Search
            q_vec = get_query_embedding(question)
            
            # Retrieve using Numpy
            retrieved_chunks = find_best_chunks(q_vec, doc_vectors, chunks, k=5)
            context = "\n\n".join(retrieved_chunks)

            # Generate
            print("Bot: Thinking...")
            answer = generate_answer(context, question)
            print(f"Bot: {answer}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"[!] Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())