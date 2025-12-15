import os

# Fix OpenMP crashes on Mac
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import asyncio
import json
import uuid
import time
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core import exceptions # Import Google exceptions

# Unstructured Imports
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env")

genai.configure(api_key=API_KEY)

# LOCAL INGESTION
def load_and_chunk_locally(pdf_path: str):
    print(f"\n[+] Parsing '{pdf_path}' locally...")
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

def get_embeddings(texts):
    result = genai.embed_content(model="models/embedding-001", content=texts, task_type="retrieval_document")
    return np.array(result['embedding'])

def get_query_embedding(text):
    result = genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_query")
    return np.array(result['embedding'])

def find_best_chunks(query_vec, doc_vecs, chunks, k=5):
    dot_product = np.dot(doc_vecs, query_vec)
    top_indices = np.argsort(dot_product)[::-1][:k]
    return [chunks[i] for i in top_indices]

# ROBUST GENERATION
def generate_answer_robust(context, question, retries=3):
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
    
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        
        except exceptions.ResourceExhausted:
            print(f"    [!] Rate Limit hit. Sleeping for 10s... (Attempt {attempt+1}/{retries})")
            time.sleep(10) # Wait longer if we hit a limit
            
        except Exception as e:
            # Print the ACTUAL error so we can see it
            print(f"    [!] Error: {str(e)}")
            return "Error: Could not generate response."
            
    return "Error: Rate limit exceeded after retries."

# 5. EXTRACTION LOGIC
EXTRACTION_QUESTIONS = {
  "basics": {
    "property_address": "What is the full property address?",
    "landlord_name": "What is the name of the Landlord?",
    "tenant_names": "Who is the Tenant?",
    "lease_start_date": "What is the lease start date?",
    "lease_end_date": "What is the lease end date?"
  },
  "rent": {
    "monthly_rent_amount": "What is the monthly rent amount?",
    "rent_due_date": "When is rent due?",
    "late_fee_policy": "What is the late fee policy?"
  },
  "deposits": {
    "security_deposit": "What is the security deposit amount?",
    "refund_rules": "What are the rules for refunding the deposit?"
  },
  "rules": {
    "pets": "What is the pet policy?",
    "smoking": "What is the smoking policy?",
    "utilities": "Which utilities are paid by the tenant?"
  }
}

async def run_full_extraction(doc_vectors, chunks):
    print("\n[+] Starting Full Extraction...")
    print("    (Slowing down to respect Free Tier limits...)")
    
    results = {}

    for category, questions in EXTRACTION_QUESTIONS.items():
        results[category] = {}
        print(f"--- Processing {category} ---")
        
        for key, question in questions.items():
            print(f"    Asking: {key}...")
            
            # 1. Search
            q_vec = get_query_embedding(question)
            retrieved_chunks = find_best_chunks(q_vec, doc_vectors, chunks, k=4)
            context = "\n\n".join(retrieved_chunks)
            
            # 2. Answer (Using Robust Function)
            answer = generate_answer_robust(context, question)
            results[category][key] = answer
            
            # 3. SLEEP: 4 seconds = 15 requests per minute
            time.sleep(4.0) 

    # Save to file
    filename = f"lease_summary_{uuid.uuid4().hex[:6]}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n[+] Summary saved to: {filename}")
    print(json.dumps(results, indent=2))

# 6. RISK ANALYSIS
def analyze_risks(doc_vectors, chunks):
    print("\n[+] Scanning for Red Flags (Indemnification, Termination, Penalties)...")
    
    # Search for generic "danger" terms to get relevant context
    risk_query = "indemnification liability terminate default penalty damages"
    q_vec = get_query_embedding(risk_query)
    
    # Retrieve top 10 chunks to get a broad view
    retrieved_chunks = find_best_chunks(q_vec, doc_vectors, chunks, k=10)
    context = "\n\n".join(retrieved_chunks)
    
    prompt = f"""
    Analyze the following lease excerpts for RED FLAGS.
    Focus on:
    1. Unlimited Indemnification (Tenant pays for everything).
    2. Landlord's right to terminate without cause.
    3. Excessive late fees or penalties.
    
    Excerpts:
    {context}
    
    Report:
    """
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        response = model.generate_content(prompt)
        print("\n=== 🚩 RISK ANALYSIS REPORT ===")
        print(response.text)
        print("===============================")
    except Exception as e:
        print(f"Error analyzing risks: {e}")


# ==========================================
# MAIN APP
# ==========================================
async def main():
    print("========================================")
    print("  Local RAG (Robust Version)")
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
    doc_vectors = get_embeddings(chunks)

    # 4. Menu Loop
    while True:
        print("\n--------------------------------")
        print("[1] Chat with Document")
        print("[2] Generate Full Summary (JSON)")
        print("[3] Exit")
        choice = input("Choice: ").strip()

        if choice == "1":
            print("\n[+] Chat Mode. Type 'exit' to go back.")
            while True:
                q = input("You: ").strip()
                if q.lower() in ['exit', 'quit', 'back']: break
                if not q: continue

                q_vec = get_query_embedding(q)
                retrieved = find_best_chunks(q_vec, doc_vectors, chunks, k=5)
                ans = generate_answer_robust("\n".join(retrieved), q)
                print(f"Bot: {ans}")

        elif choice == "2":
            await run_full_extraction(doc_vectors, chunks)

        elif choice == "3":
            print("Goodbye!")
            break

if __name__ == "__main__":
    asyncio.run(main())