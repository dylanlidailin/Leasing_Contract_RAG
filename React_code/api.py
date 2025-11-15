import os
import uuid
import asyncio  # Added for parallel queries
from pathlib import Path
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- LangChain & Google Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
import google.generativeai as genai

import uvicorn

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("Please set GEMINI_API_KEY in your .env file")

# Configure the Google AI SDK
try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    raise RuntimeError(f"Failed to configure Google AI: {e}. Check your GEMINI_API_KEY.")

# Set the Gemini LLM for LangChain
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)

# --- Session Store ---
SESSION_STORE = {}

# ---- FastAPI App ----
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Chain Builders ----

def build_rag_chain(docs: list):
    """Builds the RAG (chat) chain."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    pages = splitter.split_documents(docs)
    
    vectorstore = FAISS.from_documents(pages, embeddings)
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        "Use the following context to answer the question:\n\n{context}\n\nQuestion: {input}\n\nAnswer:"
    )

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return rag_chain

# ---- Extraction Questions ----
# Defined all questions needed to fill the JSON
EXTRACTION_QUESTIONS = {
  "basics": {
    "property_address": "What is the full property address for the lease?",
    "landlord_name": "What is the name of the Landlord or Lessor?",
    "tenant_names": "What are the names of all Tenants or Lessees listed? Respond with a comma-separated list.",
    "lease_start_date": "What is the lease start date? (YYYY-MM-DD)",
    "lease_end_date": "What is the lease end date? (YYYY-MM-DD)",
    "occupancy_limit": "What is the occupancy limit (number of people)?"
  },
  "rent_and_payments": {
    "monthly_rent_amount": "What is the monthly rent amount in dollars? (e.g., 1500.00)",
    "rent_due_day": "What day of the month is rent due?",
    "grace_period_days": "How many days is the grace period for late rent?",
    "late_fee_policy": "What is the late fee policy?",
    "accepted_payment_methods": "What are the accepted payment methods for rent? (e.g., check, online portal)"
  },
  "deposits_and_fees": {
    "security_deposit_amount": "What is the security deposit amount?",
    "refundability_rules": "What are the rules for refunding the security deposit?",
    "non_refundable_fees": "What are the non-refundable fees? (e.g., cleaning fee)",
    "move_out_notice_period_days": "What is the move-out notice period in days?"
  },
  "utilities": {
    "utilities_included": "What utilities are included in the rent? (e.g., water, trash)",
    "utilities_tenant_responsible_for": "What utilities is the tenant responsible for? (e.g., electricity, gas)"
  },
  "rules": {
    "smoking_policy": "What is the smoking policy?",
    "pet_policy": "What is the pet policy? (e.g., allowed with fee, not allowed)",
    "subletting_policy": "What is the subletting policy?",
    "parking_policy": "What is the parking policy?"
  },
  "maintenance_and_termination": {
    "landlord_maintenance_responsibilities": "What are the landlord's maintenance responsibilities?",
    "entry_notice_period_hours": "What is the notice period in hours for landlord entry?",
    "early_termination_terms": "What are the terms for early termination of the lease?",
    "renewal_holdover_policy": "What is the policy for renewal or holdover?",
    "insurance_requirement": "Is renter's insurance required?"
  }
}

# ---- API Models ----
class Query(BaseModel):
    question: str
    session_id: str

class SessionQuery(BaseModel):
    session_id: str

# ---- API Endpoints ----
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    This endpoint sets up the session for the chatbot and extraction features.
    It builds the RAG chain and stores it in the session.
    """
    try:
        suffix = Path(file.filename).suffix or ".pdf"
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp.flush()
            path = tmp.name

        loader = PyPDFLoader(path)
        docs = loader.load()
        if not docs:
            raise HTTPException(status_code=400, detail="Failed to load any documents from PDF.")

        # 1. Build RAG chain
        rag_chain = build_rag_chain(docs)
        if rag_chain is None:
            raise HTTPException(status_code=400, detail="Failed to build retrieval chain")

        # 2. Store chain in session (REMOVED raw docs to save memory)
        session_id = str(uuid.uuid4())
        SESSION_STORE[session_id] = {
            "rag_chain": rag_chain,
            # "docs": docs  <- Removed
        }

        # 3. Return session ID
        return {
            "status": "Uploaded",
            "filename": file.filename,
            "session_id": session_id
        }

    except Exception as e:
        print(f"[PDF analysis error] {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'path' in locals() and Path(path).exists():
            os.remove(path)

@app.post("/ask")
async def ask(q: Query):
    """Handles chatbot questions."""
    session_data = SESSION_STORE.get(q.session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    chain = session_data.get("rag_chain")
    if not chain:
        raise HTTPException(status_code=400, detail="RAG chain not found in session.")

    try:
        response = await chain.ainvoke({"input": q.question})
        answer = response.get("answer") or "Could not find an answer."
    except Exception as e:
        print(f"[RAG Error] {e}")
        answer = f"Error processing request: {e}"

    return {"answer": answer}

# --- Helper for new get_summary ---
async def run_query(chain, question):
    """Runs a single RAG query and returns the answer."""
    try:
        response = await chain.ainvoke({"input": question})
        answer = response.get("answer")
        # Simple cleanup for "null" or "N/A" type responses
        if answer is None or "not found" in answer.lower() or "n/a" in answer.lower():
            return "null"
        return answer
    except Exception as e:
        print(f"Error during query '{question}': {e}")
        return "null"


@app.post("/get_summary")
async def get_summary(q: SessionQuery):
    """Handles JSON summary extraction using parallel RAG queries."""
    session_data = SESSION_STORE.get(q.session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    chain = session_data.get("rag_chain")
    if not chain:
        raise HTTPException(status_code=400, detail="RAG chain not found in session.")

    try:
        # 1. Create a list of all async tasks to run
        tasks = []
        # This dict will help us map results back later
        query_to_key_map = {} 

        for category, fields in EXTRACTION_QUESTIONS.items():
            if category not in fields: # Create nested dict for results
                fields[category] = {}
            for key, question in fields.items():
                tasks.append(run_query(chain, question))
                # Store a "path" to the key for easy re-assembly
                query_to_key_map[question] = (category, key) 

        # 2. Run all tasks in parallel
        print(f"Running {len(tasks)} extraction queries in parallel for session {q.session_id}...")
        results = await asyncio.gather(*tasks)
        print("...Queries complete.")

        # 3. Assemble the final JSON
        extracted_data = {}
        all_questions = list(query_to_key_map.keys())
        
        for i in range(len(results)):
            question = all_questions[i]
            answer = results[i]
            category, key = query_to_key_map[question]
            
            if category not in extracted_data:
                extracted_data[category] = {}
            
            # Here you could add logic to parse types (e.g., float for rent)
            # For now, we'll just assign the string answer.
            extracted_data[category][key] = answer

        return {
            "status": "Extracted",
            "data": extracted_data
        }
        
    except Exception as e:
        print(f"[Extraction Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/healthz")
async def health():
    return {"status": "ok"}

# ---- Run locally ----
if __name__ == "__main__":
    import uvicorn
    print("Starting server on http://localhost:8000")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)