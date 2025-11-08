import os
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

import logging
import uvicorn

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in your .env file")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=API_KEY)

SESSION_STORE = {}

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

# ---- RAG Chain Builder ----
def build_chain_from_pdf(path: str):
    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    pages = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(pages, embeddings)
    retriever = vectorstore.as_retriever()

    # Define prompt templates
    prompt = ChatPromptTemplate.from_template(
        "Use the following context to answer the question:\n\n{context}\n\nQuestion: {input}"
    )

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return rag_chain

# ---- API Endpoints ----
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        suffix = Path(file.filename).suffix or ".pdf"
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp.flush()
            path = tmp.name

        chain = build_chain_from_pdf(path)
        if chain is None:
            raise HTTPException(status_code=400, detail="Failed to build retrieval chain")

        session_id = str(uuid.uuid4())
        SESSION_STORE[session_id] = chain

        return {
            "status": "Uploaded",
            "filename": file.filename,
            "session_id": session_id
        }

    except Exception as e:
        print("[PDF analysis error]", str(e))
        raise HTTPException(status_code=500, detail=str(e))

class Query(BaseModel):
    question: str
    session_id: str

@app.post("/ask")
async def ask(q: Query):
    chain = SESSION_STORE.get(q.session_id)

    if chain is None:
        return {"answer": "Session not found or expired."}

    try:
        response = chain.invoke({"input": q.question})
        answer = response.get("answer") or response
    except Exception as e:
        answer = f"RAG error: {str(e)}"

    return {"answer": answer}

@app.get("/healthz")
async def health():
    return {"status": "ok"}

# ---- Run locally ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)