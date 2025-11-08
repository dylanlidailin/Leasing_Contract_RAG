import os
import uuid
import json  # Added import
from pathlib import Path
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

import logging
import uvicorn

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in your .env file")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=API_KEY)

# ---- FastAPI App ----
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_file = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_file.read_text())

# ---- RAG Chain Builder (Now an Extraction Chain) ----
def build_chain_from_pdf(path: str):
    loader = PyPDFLoader(path)
    docs = loader.load()

    # This new prompt hard-codes your categories for extraction
    extraction_prompt = """
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

    prompt = ChatPromptTemplate.from_template(extraction_prompt)
    
    # We use create_stuff_documents_chain to pass all docs
    extraction_chain = create_stuff_documents_chain(llm, prompt)
    
    # Return the chain AND the loaded documents
    return extraction_chain, docs

# ---- API Endpoints ----
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        suffix = Path(file.filename).suffix or ".pdf"
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp.flush()
            path = tmp.name

        # 1. Build the chain and get the docs
        extraction_chain, docs = build_chain_from_pdf(path)
        if extraction_chain is None:
            raise HTTPException(status_code=400, detail="Failed to build extraction chain")

        # 2. Run the extraction immediately
        # We pass the loaded docs as the 'context'
        response_str = extraction_chain.invoke({
            "context": docs,
            "input": "Extract all data."  # 'input' is required but not used by our prompt
        })

        # 3. Parse the LLM's JSON response
        try:
            extracted_data = json.loads(response_str)
        except json.JSONDecodeError:
            # Handle cases where the LLM's output is not perfect JSON
            print("[LLM Response Error] Output was not valid JSON:")
            print(response_str)
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to parse LLM response as JSON. Raw: {response_str}"
            )

        # 4. Return the structured data
        return {
            "status": "Extracted",
            "filename": file.filename,
            "data": extracted_data
        }

    except Exception as e:
        print("[PDF analysis error]", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# --- Chat endpoints removed as they are replaced by extraction ---
# The 'SESSION_STORE', 'Query' model, and '/ask' endpoint
# are no longer needed for this workflow.

@app.get("/healthz")
async def health():
    return {"status": "ok"}

# ---- Run locally ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True)