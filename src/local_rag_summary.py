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
from google.api_core import exceptions

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
    #text_chunks = [chunk.text for chunk in chunks]
    #print(f"[+] Created {len(text_chunks)} chunks.")
    records = []
    
    for i, chunk in enumerate(chunks):
        records.append({
            "chunk_id": f"chunk_{i}",
            "text": chunk.text,
            "page": getattr(chunk.metadata, "page_number", None),
            "section": getattr(chunk.metadata, "section_title", None),
            #"page": chunk.metadata.get("page_number"),
            #"section": chunk.metadata.get("section_title"),
        })
    return records

def get_embeddings(texts):
    result = genai.embed_content(model="models/embedding-001", content=texts, task_type="retrieval_document")
    return np.array(result['embedding'])

def get_query_embedding(text):
    result = genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_query")
    return np.array(result['embedding'])

def find_best_chunks(query_vec, doc_vecs, records, k=5):
    #dot_product = np.dot(doc_vecs, query_vec)
    scores = np.dot(doc_vecs, query_vec)
    top_indices = np.argsort(scores)[::-1][:k]
    #return [chunks[i] for i in top_indices]
    return [{
        "text": records[i]["text"],
        "page": records[i]["page"],
        "section": records[i]["section"],
        "score": float(scores[i]),
        "chunk_id": records[i]["chunk_id"]
    } for i in top_indices]

# ROBUST GENERATION
def generate_answer_robust(context, question, retries=3):
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    You are an expert lease analyst.
    RULES:
    1. Every factual claim must be supported by a verbatim quote.
    2. After each quote, include the page number.
    3. If the excerpts do not fully support an answer, say: "Not specified in the lease."
    4. If the context shows a label like "Tenant:" but no name, state that it is blank.
    5. 'Buyer/Tenant' refers to the Tenant.
    
    Excerpts:
    {context}
    
    Question: {question}
    Answer:
    Citations:
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
            #context = "\n\n".join(retrieved_chunks)
            def format_evidence(evidence_list):
                blocks = []
                for i, e in enumerate(evidence_list, 1):
                    blocks.append(
                        f"[{i}] Page {e['page']} ({e['section']}):\n\"{e['text']}\""
                    )
                return "\n\n".join(blocks)
            
            context = format_evidence(retrieved_chunks)
            
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
    #context = "\n\n".join(retrieved_chunks)
    context_blocks = []
    for chunk in retrieved_chunks:
        text_content = chunk['text']
        page_num = chunk['page']

        formatted_string = f"Page {page_num or '?'}] {text_content}"

        context_blocks.append(formatted_string)
    context = "\n\n".join(context_blocks)

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
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    try:
        response = model.generate_content(prompt)
        print("\n=== 🚩 RISK ANALYSIS REPORT ===")
        print(response.text)
        print("===============================")
    except Exception as e:
        print(f"Error analyzing risks: {e}")


# MAIN APP
async def main():
    print("========================================")
    print("  Leasing RAG model")
    print("========================================")
    
    # 1. Input
    while True:
        pdf_path = input("\nEnter PDF path: ").strip()
        if os.path.exists(pdf_path) and pdf_path.lower().endswith('.pdf'):
            break
        print("[!] File not found or not a PDF.")

    # 2. Ingest
    try:
        records = load_and_chunk_locally(pdf_path)
        texts = [r["text"] for r in records]
    except Exception as e:
        print(f"[!] Parsing Error: {e}")
        return

    # 3. Embed
    print("[+] Embedding chunks...")
    doc_vectors = get_embeddings(texts)

    # 4. Menu Loop
    while True:
        print("\n--------------------------------")
        print("[1] Chat with Document")
        print("[2] Generate Full Summary (JSON)")
        print("[3] Risk Analysis (Red Flags)")
        print("[4] Exit")                       
        choice = input("Choice: ").strip()

        if choice == "1":
            print("\n[+] Chat Mode. Type 'exit' to go back.")
            while True:
                q = input("You: ").strip()
                if q.lower() in ['exit', 'quit', 'back']: break
                if not q: continue

                q_vec = get_query_embedding(q)
                # This returns a list of DICTIONARIES now
                retrieved = find_best_chunks(q_vec, doc_vectors, records, k=5)
                
                # We loop through the dictionaries and format them into a string just like we did for the risk analysis.
                formatted_context = "\n\n".join([
                    f"[Page {r['page'] or '?'}] {r['text']}" 
                    for r in retrieved
                ])
                
                # Now we send the formatted STRING to the AI
                ans = generate_answer_robust(formatted_context, q)
                print(f"Bot: {ans}")

        elif choice == "2":
            await run_full_extraction(doc_vectors, records)

        elif choice == "3":
            analyze_risks(doc_vectors, records)

        elif choice == "4":
            print("Goodbye!")
            break

if __name__ == "__main__":
    asyncio.run(main())