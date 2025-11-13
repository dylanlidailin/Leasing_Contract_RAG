import os
import uuid
import json  # Import for parsing JSON
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
        "Use the following context to answer the question:\n\n{context}\n\nQuestion: {input}"
    )

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return rag_chain

def get_extraction_chain():
    """Builds the JSON extraction chain."""
    extraction_prompt_template = """
You are an expert legal assistant specializing in parsing lease agreements.
Use the following context from a lease agreement to extract the requested information.
Your output MUST be a single, valid JSON object. Do not add any text before or after the JSON.
Use "null" for any fields you cannot find in the context.

Context:
{context}

Based on the context, extract the following information. Use the keys provided.

{{
  "basics": {{
    "property_address": "...",
    "landlord_name": "...",
    "tenant_names": ["..."],
    "lease_start_date": "YYYY-MM-DD",
    "lease_end_date": "YYYY-MM-DD",
    "occupancy_limit": 0
  }},
  "rent_and_payments": {{
    "monthly_rent_amount": 0.0,
    "rent_due_day": "...",
    "grace_period_days": 0,
    "late_fee_policy": "...",
    "accepted_payment_methods": ["..."]
  }},
  "deposits_and_fees": {{
    "security_deposit_amount": 0.0,
    "refundability_rules": "...",
    "non_refundable_fees": ["..."],
    "move_out_notice_period_days": 0
  }},
  "utilities": {{
    "utilities_included": ["..."],
    "utilities_tenant_responsible_for": ["..."]
  }},
  "rules": {{
    "smoking_policy": "...",
    "pet_policy": "...",
    "subletting_policy": "...",
    "parking_policy": "..."
  }},
  "maintenance_and_termination": {{
    "landlord_maintenance_responsibilities": "...",
    "entry_notice_period_hours": 0,
    "early_termination_terms": "...",
    "renewal_holdover_policy": "...",
    "insurance_requirement": "..."
  }}
}}
"""
    prompt = ChatPromptTemplate.from_template(extraction_prompt_template)
    extraction_chain = create_stuff_documents_chain(llm, prompt)
    return extraction_chain

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
    This endpoint sets up the session for both the chatbot feature and the full-document extraction.
    It builds the RAG chain and stores the docs.
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

        # 2. Store chain AND docs in session
        session_id = str(uuid.uuid4())
        SESSION_STORE[session_id] = {
            "rag_chain": rag_chain,
            "docs": docs
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

@app.post("/get_summary")
async def get_summary(q: SessionQuery):
    """Handles JSON summary extraction."""
    session_data = SESSION_STORE.get(q.session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    docs = session_data.get("docs")
    if not docs:
        raise HTTPException(status_code=400, detail="Documents not found in session.")

    try:
        extraction_chain = get_extraction_chain()
        response_str = await extraction_chain.ainvoke({
            "context": docs,
            "input": "Extract all data." 
        })
        
        # Clean up potential markdown code fences from LLM response
        if response_str.strip().startswith("```json"):
            response_str = response_str.strip()[7:-3].strip()
        
        extracted_data = json.loads(response_str)
        
        return {
            "status": "Extracted",
            "data": extracted_data
        }
        
    except json.JSONDecodeError:
        print("[LLM Response Error] Output was not valid JSON:")
        print(response_str)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to parse LLM response as JSON. Raw: {response_str}"
        )
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