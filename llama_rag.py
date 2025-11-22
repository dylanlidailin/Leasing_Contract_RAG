import os
import asyncio
import numpy as np
import faiss
import google.generativeai as genai
from dotenv import load_dotenv
from llama_parse import LlamaParse

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
LLAMA_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

if not API_KEY or not LLAMA_KEY:
    raise ValueError("Missing API Keys")

genai.configure(api_key=API_KEY)

# --- 1. Native Text Splitter ---
def simple_text_splitter(text, chunk_size=1000, overlap=100):
    """Splits text into chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        # Move forward, keeping the overlap
        start += (chunk_size - overlap)
    return chunks

# --- 2. Native Embeddings ---
def get_embeddings(texts):
    """
    Batch embeds text using Google's SDK directly.
    Returns a numpy array for FAISS.
    """
    # Google's batch embedding API
    result = genai.embed_content(
        model="models/embedding-001",
        content=texts,
        task_type="retrieval_document"
    )

    # Convert list of vectors to float32 numpy array for FAISS
    return np.array(result['embedding'], dtype='float32')

def get_query_embedding(text):
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_query"
    )
    return np.array([result['embedding']], dtype='float32')

# --- 3. Native LLM Chat ---
def generate_answer(context, question):
    """Generates an answer using the Gemini model directly."""
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    You are a helpful lease assistant.
    Context: {context}
    
    Question: {question}
    Answer:
    """
    response = model.generate_content(prompt)
    return response.text

# --- Main Logic ---
async def main():
    pdf_path = "Data/wire_fraud" #for demo
    
    # A. Load (LlamaParse)
    print("[+] Parsing PDF...")
    parser = LlamaParse(api_key=LLAMA_KEY, result_type="markdown")
    documents = await parser.aload_data(pdf_path)
    full_text = "\n".join([doc.text for doc in documents])

    # B. Split
    print("[+] Splitting text...")
    chunks = simple_text_splitter(full_text)
    print(f"    Created {len(chunks)} chunks.")

    # C. Embed & Index (FAISS)
    print("[+] Embedding and Indexing...")
    embeddings = get_embeddings(chunks)
    dimension = embeddings.shape[1]
    
    # Create FAISS index (L2 distance)
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # D. Chat Loop
    while True:
        question = input("\nYou: ")
        if question.lower() in ['exit', 'quit']: break

        # 1. Embed Question
        q_vec = get_query_embedding(question)

        # 2. Search (Retrieve Top 4)
        distances, indices = index.search(q_vec, k=4)
        
        # 3. Build Context
        retrieved_chunks = [chunks[i] for i in indices[0]]
        context = "\n\n".join(retrieved_chunks)

        # 4. Generate Answer
        print("Bot: Thinking...")
        answer = generate_answer(context, question)
        print(f"Bot: {answer}")

if __name__ == "__main__":
    asyncio.run(main())