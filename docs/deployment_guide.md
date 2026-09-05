# Deployment Guide

## 1. Local Setup (Prototype / Demo)

```bash
# 1. Clone / unzip the project, then create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key (OpenAI shown; see Azure section below to switch)
export OPENAI_API_KEY="sk-..."          # macOS/Linux
setx OPENAI_API_KEY "sk-..."            # Windows

# 4. Ingest the sample documents (or point --dir at your own folder)
python -m ingestion.ingest --dir data/sample_docs --reset

# 5. Launch the chatbot
streamlit run app.py

# 6. (Optional) Run evaluation and view the dashboard
python -m eval.evaluate
streamlit run eval/dashboard.py
```

The app opens at `http://localhost:8501`. The evaluation dashboard runs on
a separate port if launched simultaneously (Streamlit will prompt to use
`8502`).

## 2. Ingesting Your Own Documents
1. Drop PDF/DOCX/PPTX/TXT files into a folder (e.g. `data/my_docs/`).
2. Optionally add a metadata header to the top of TXT/MD files (or the
   first page of PDFs) for cleaner filtering:
   ```
   TITLE: Remote Work Policy
   DEPARTMENT: HR
   SENSITIVITY: Internal
   VERSION: 1.2
   EFFECTIVE_DATE: 2026-03-01
   ```
   Documents without a header default to `department=General`,
   `sensitivity=Internal`, `version=1.0`, and the filename as title — the
   pipeline never fails on missing metadata, it just degrades gracefully.
3. Run: `python -m ingestion.ingest --dir data/my_docs --reset`
   (omit `--reset` to add to the existing index instead of replacing it).

## 3. Switching to Azure OpenAI
Set these environment variables instead of `OPENAI_API_KEY`:
```bash
export RAG_PROVIDER="azure_openai"
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com"
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_API_VERSION="2024-08-01-preview"
export AZURE_EMBEDDING_DEPLOYMENT="<your-embedding-deployment-name>"
export AZURE_CHAT_DEPLOYMENT="<your-chat-deployment-name>"
```
No code changes are required — `rag/llm_client.py` reads `RAG_PROVIDER` and
routes accordingly.

## 4. Moving to a Production Vector Store
`rag/vector_store.py` is the only file that talks to Chroma. To move to
Pinecone, Azure AI Search, or pgvector:
1. Implement the same three methods (`upsert_chunks`, `query`,
   `get_all_documents_summary`) against the new backend.
2. Point `rag/retriever.py` and `app.py` at the new class — no other file
   needs to change.
3. For Pinecone specifically: create an index with dimension 1536 (matching
   `text-embedding-3-small`), metric `cosine`, and store metadata as
   Pinecone metadata fields (department, sensitivity, source, page,
   version) to preserve filtering.

## 5. Cloud Deployment Options
- **Streamlit Community Cloud**: simplest path for an internal demo — push
  to a private GitHub repo, connect, set secrets (`OPENAI_API_KEY`, etc.)
  in the Streamlit Cloud secrets manager.
- **Azure App Service / Container Apps**: containerize with a Dockerfile
  (`streamlit run app.py --server.port 8000 --server.address 0.0.0.0`),
  push to Azure Container Registry, deploy behind Azure AD auth for SSO.
- **Internal Kubernetes**: same container, exposed via internal ingress,
  vector store either co-located (Chroma with a persistent volume) or
  pointed at a managed service (Azure AI Search / Pinecone) for HA.

### Example Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 6. Production Hardening Checklist
- [ ] Replace demo metadata-filter access control with real identity-aware
      filtering (Entra ID groups / SharePoint permissions resolved per
      user at query time).
- [ ] Move ChromaDB to a managed/HA vector store, or run Chroma server mode
      with a persistent, backed-up volume.
- [ ] Add request/response audit logging (query, retrieved chunk IDs,
      answer, user identity, timestamp) for compliance.
- [ ] Add rate limiting and cost monitoring on LLM/embedding API calls.
- [ ] Schedule incremental re-ingestion (e.g., nightly) for documents that
      change, using file hash or `last_modified` to skip unchanged files.
- [ ] Add automated re-runs of `eval/evaluate.py` in CI against a growing
      labeled test set to catch retrieval/answer quality regressions.
- [ ] Review and rotate API keys via a secrets manager (Azure Key Vault,
      AWS Secrets Manager) rather than plain environment variables.

## 7. Troubleshooting
| Symptom | Likely Cause | Fix |
|---|---|---|
| `store.count() == 0` in the UI | Ingestion not run, or wrong `CHROMA_DIR` | Run `python -m ingestion.ingest` |
| Empty/garbled PDF text | Scanned (image-only) PDF | Add OCR (e.g., `pytesseract`) as a pre-processing step before `load_pdf` |
| Cross-encoder reranker slow on first call | Model download on first use | Pre-warm by running ingestion/eval once after deployment; model caches locally afterward |
| Answers ignore department filter | `where` clause built from empty selection | Filters are additive — no selection means "search everything," by design |
