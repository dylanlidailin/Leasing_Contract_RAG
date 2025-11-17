import os
import uuid
import asyncio
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

# --- LangChain & Google Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
import google.generativeai as genai

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


# ---- Chain Builders ----
def build_rag_chain(docs: list):
    """Builds the RAG (chat) chain."""
    print("[+] Splitting and indexing document chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    pages = splitter.split_documents(docs)
    
    # Store a json file of the splitted chunks for debugging
    with open("debug_splitted_chunks.json", "w") as f:
        json.dump([page.page_content for page in pages], f, indent=2)
    print(f"[+] Document split into {len(pages)} chunks.")

    vectorstore = FAISS.from_documents(pages, embeddings)
    retriever = vectorstore.as_retriever()

    # This is the improved prompt to handle placeholders
    # We are making RULE 2 much more specific to handle the
    # 'Buyer/Tenant' and 'Seller/Landlord' labels.
    system_prompt = """
    You are an expert assistant for analyzing documents. 
    Use the following context to answer the question clearly and directly. 
    Your priority is to find specific, filled-in information.

    --- RULES ---
    
    RULE 1 (IGNORE PLACEHOLDERS): 
    If the context contains both a real, filled-in value (like a specific address or name)
    AND a generic placeholder (like '(Property)' or '(Tenant)'),
    ALWAYS prioritize the real, filled-in value and IGNORE the placeholder.
    
    RULE 2 (PAY ATTENTION TO ROLES): 
    Pay close attention to labels next to names.
    - The label 'Buyer/Tenant' refers to the 'Tenant' or 'Buyer'.
    - The label 'Seller/Landlord' refers to the 'Seller' or 'Landlord'.
    
    If the question asks for the 'Tenant' or 'Buyer', you MUST only provide the name(s) associated with the 'Buyer/Tenant' label.
    If the question asks for the 'Seller' or 'Landlord', you MUST only provide the name(s) associated with the 'Seller/Landlord' label.
    Do not mix them up or get them inverted.

    RULE 3 (HOW TO ANSWER WHEN EMPTY):
    If the requested information (like a name, date, or amount) is not found in the context or the relevant section is blank,
    do not just say 'None' or 'N/A'.
    Respond with a clear sentence, such as: 'That information is not specified in the document.' or 'The section for that topic appears to be blank.'

    ---
    
    Answer only with the requested information based on these rules.

    Context:
    {context}

    Question: {input}

    Answer:
    """

    prompt = ChatPromptTemplate.from_template(system_prompt)

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    print("[+] RAG chain built successfully.")
    return rag_chain

# ---- Extraction Questions ----
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

# ---- RAG Query Helper ----
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

# ---- CLI-Specific Functions ----

async def get_summary_from_chain(chain):
    """
    This function runs all extraction queries in parallel
    and prints the final assembled JSON.
    """
    print("\n[+] Generating full document summary...")
    print("    This may take up to a minute. Please wait.")
    tasks = []
    query_to_key_map = {} 

    # 1. Create all the async tasks
    for category, fields in EXTRACTION_QUESTIONS.items():
        for key, question in fields.items():
            tasks.append(run_query(chain, question))
            # Store a "path" to the key for easy re-assembly
            query_to_key_map[question] = (category, key) 

    # 2. Run all tasks in parallel
    print(f"    Running {len(tasks)} extraction queries in parallel...")
    results = await asyncio.gather(*tasks)
    print("[+] Queries complete. Assembling JSON...")
    
    # 3. Assemble the final JSON
    extracted_data = {}
    all_questions = list(query_to_key_map.keys())
    
    for i in range(len(results)):
        question = all_questions[i]
        answer = results[i]
        category, key = query_to_key_map[question]
        
        if category not in extracted_data:
            extracted_data[category] = {}
        
        extracted_data[category][key] = answer

    # Store the answer as a json file to the repo
    output_filename = f"lease_summary_{uuid.uuid4().hex[:8]}.json"
    with open(output_filename, "w") as f:
        json.dump(extracted_data, f, indent=2)
    print(f"[+] Summary saved to '{output_filename}'.")

    # 4. Print the final, assembled JSON
    print("\n[+] Summary Complete!\n")
    print(json.dumps(extracted_data, indent=2))


async def run_chatbot_loop(chain):
    """
    This function runs a continuous loop to ask for user questions.
    """
    print("\n[+] Entering Chatbot Mode.")
    print("    Ask a question. Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            question = input("You: ")
            if question.lower() in ['quit', 'exit']:
                break
            
            if not question.strip():
                continue
                
            # Give feedback while waiting
            print("    Bot: Thinking...")
            answer = await run_query(chain, question)
            
            print(f"\nBot: {answer}\n")

        except (EOFError, KeyboardInterrupt):
            break
            
    print("\n[+] Exiting Chatbot Mode.")


async def main():
    """
    The main entry point for the CLI application.
    """
    print("========================================")
    print("  Lease Agreement Analyzer CLI")
    print("========================================\n")
    
    # 1. Get PDF path
    pdf_path_str = ""
    while True:
        pdf_path_str = input("Please enter the path to your PDF lease agreement: ").strip()
        pdf_path = Path(pdf_path_str)
        if pdf_path.exists() and pdf_path.suffix.lower() == '.pdf':
            break
        print("[!] Invalid path. Please provide a valid path to a .pdf file.\n")
    
    # 2. Load and build the chain
    rag_chain = None
    try:
        print(f"\n[+] Loading and processing '{pdf_path.name}'...")
        loader = PyPDFLoader(pdf_path_str)
        docs = loader.load()
        if not docs:
            print("[!] Failed to load any documents from PDF.")
            return
        
        rag_chain = build_rag_chain(docs)
    except Exception as e:
        print(f"[!] An error occurred while processing the PDF: {e}")
        return

    # 3. Present the choice
    while True:
        print("\n----------------------------------------")
        print("What would you like to do?")
        print("  [1] Ask questions (Chatbot Mode)")
        print("  [2] Get full document summary (JSON)")
        print("  [3] Quit")
        choice = input("Choose an option (1, 2, or 3): ").strip()
        
        if choice == "1":
            await run_chatbot_loop(rag_chain)
        elif choice == "2":
            await get_summary_from_chain(rag_chain)
        elif choice == "3":
            print("\nGoodbye!")
            break
        else:
            print("[!] Invalid choice. Please enter 1, 2, or 3.")


# ---- Run CLI ----
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[!] Application interrupted. Exiting.")