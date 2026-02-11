# Leasing Contract RAG

A **Retrieval-Augmented Generation (RAG)** application for parsing and analyzing residential leasing contracts. Upload a lease PDF, and the system chunks it, embeds it with Google’s Gemini embeddings, and lets you chat with the document, generate structured summaries, or run a risk analysis.

## Features

- **Local PDF ingestion** — Uses [Unstructured](https://unstructured.io/) to parse PDFs and chunk by title/section (no external document API required for ingestion).
- **Semantic search** — Gemini `embedding-001` for document and query embeddings; cosine similarity for retrieving the most relevant chunks.
- **Chat with document** — Ask free-form questions; answers are grounded in retrieved excerpts with page references.
- **Structured extraction** — One-click “Full Summary” to extract a fixed set of fields (parties, dates, rent, deposits, pet/smoking/utilities) into a JSON file.
- **Risk analysis** — Scans the lease for red flags: broad indemnification, termination without cause, and excessive fees/penalties.

## Requirements

- **Python 3.10+**
- **Google Gemini API key** (for embeddings and generation)

The core script uses:

- `unstructured` — PDF parsing and chunking  
- `google-generativeai` — embeddings and Gemini 2.5 Flash  
- `numpy` — vector similarity  
- `python-dotenv` — loading `GEMINI_API_KEY` from `.env`

See `requirements.txt` for the full dependency list (including optional LangChain, LlamaParse, FAISS, and FastAPI).

## Setup

1. **Clone and install**

   ```bash
   cd Leasing_contract_RAG-1
   pip install -r requirements.txt
   ```

   If you only run the core script, ensure you have at least:

   ```bash
   pip install google-generativeai python-dotenv numpy unstructured
   ```

2. **API key**

   Create a `.env` file in the project root:

   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

   Get a key from [Google AI Studio](https://aistudio.google.com/apikey).

## Usage

Run the main script:

```bash
python src/local_rag_summary.py
```

1. **Enter PDF path** when prompted (e.g. `Data/Res. Lease_Month-to-Month Rental Agmt  17 Starflower 02.26 (1).pdf`).
2. Wait for parsing and embedding to finish.
3. Choose an option from the menu:
   - **1 — Chat with Document** — Ask questions; type `exit` to return to the menu.
   - **2 — Generate Full Summary (JSON)** — Extracts all defined fields and saves a timestamped JSON file (e.g. `lease_summary_abc123.json`).
   - **3 — Risk Analysis** — Prints a short report on indemnification, termination, and penalty red flags.
   - **4 — Exit**

## Extracted Fields (Full Summary)

| Category   | Fields |
|-----------|--------|
| **Basics** | Property address, Landlord name, Tenant name(s), Lease start/end dates |
| **Rent**  | Monthly rent, Rent due date, Late fee policy |
| **Deposits** | Security deposit amount, Refund rules |
| **Rules** | Pet policy, Smoking policy, Utilities paid by tenant |

Answers are citation-based: the model is instructed to support claims with verbatim quotes and page numbers, and to say “Not specified in the lease” when the excerpts don’t support an answer.

## Project structure

```
Leasing_contract_RAG-1/
├── src/
│   └── local_rag_summary.py   # Core RAG + summary + risk script
├── Data/                      # Sample lease PDFs
├── requirements.txt
├── .env                       # GEMINI_API_KEY (create this)
└── README.md
```

## Notes

- **Rate limits** — Full summary runs with a 4-second delay between Gemini calls to stay within free-tier limits; the script also retries on rate-limit errors.
- **Mac** — The script sets `KMP_DUPLICATE_LIB_OK=TRUE` to avoid OpenMP-related crashes with Unstructured/numpy on macOS.

## License

Use and modify as needed for your environment.