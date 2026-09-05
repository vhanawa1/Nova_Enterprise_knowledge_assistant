# Enterprise Knowledge Assistant — RAG Architecture

## 1. Problem Recap
Enterprise knowledge (policies, SOPs, manuals, reports) is scattered across
SharePoint, drives, and portals. Keyword search returns file lists, not
answers. This solution converts that corpus into a conversational,
source-cited, access-aware assistant.

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INGESTION PIPELINE                         │
│                                                                       │
│  SharePoint/Drive/PDF/DOCX/PPTX                                     │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────┐    ┌───────────────┐    ┌─────────────────────┐   │
│  │   Loaders    │───▶│   Chunking    │───▶│  Embedding (OpenAI/  │   │
│  │ (per format) │    │ (recursive,   │    │  Azure OpenAI)       │   │
│  │ + metadata   │    │  800/150 ovl) │    │  text-embedding-3-sm │   │
│  │  extraction  │    │               │    │                      │   │
│  └──────────────┘    └───────────────┘    └──────────┬───────────┘   │
│                                                        ▼               │
│                                          ┌──────────────────────┐    │
│                                          │   ChromaDB (vector    │    │
│                                          │   store, persisted,   │    │
│                                          │   metadata-indexed)   │    │
│                                          └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          QUERY PIPELINE                              │
│                                                                       │
│  User question (+ chat history)                                     │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────┐   ┌───────────────┐   ┌─────────────────────┐     │
│  │  Metadata    │──▶│ Dense vector  │──▶│  Hybrid fusion        │     │
│  │  filter      │   │ search top-12 │   │  (BM25 + cosine)      │     │
│  │ (dept/       │   │ (ChromaDB)    │   │                       │     │
│  │  sensitivity)│   │               │   └──────────┬───────────┘     │
│  └──────────────┘   └───────────────┘              ▼                 │
│                                          ┌──────────────────────┐    │
│                                          │  Cross-encoder rerank │    │
│                                          │  → top 4 chunks       │    │
│                                          └──────────┬───────────┘    │
│                                                      ▼                 │
│                                          ┌──────────────────────┐    │
│                                          │  LLM generation       │    │
│                                          │  (grounded prompt,    │    │
│                                          │   citations, history) │    │
│                                          └──────────┬───────────┘    │
│                                                      ▼                 │
│                              Answer + Citations [S1][S2] + Confidence │
└─────────────────────────────────────────────────────────────────────┘
```

## 3. Component Details

### 3.1 Ingestion
- **Loaders** (`ingestion/loaders.py`): format-specific extractors for PDF
  (pypdf, page-level), DOCX (python-docx, incl. tables), PPTX (python-pptx,
  slide-level), TXT/MD. Each also parses a lightweight metadata header
  (title, department, sensitivity, version, effective date) — mirroring how
  a real pipeline would pull SharePoint columns or document properties via
  Microsoft Graph API.
- **Chunking** (`ingestion/chunking.py`): recursive splitter — paragraph →
  sentence → hard cutoff — with 800-char chunks and 150-char overlap. Every
  chunk keeps full lineage: source file, page/slide number, department,
  sensitivity, version, and a unique chunk_id, enabling exact citation and
  traceability back to the original document location.
- **Embedding**: OpenAI `text-embedding-3-small` (1536-dim) via a
  provider-agnostic client (`rag/llm_client.py`) that swaps to Azure OpenAI
  by changing one config value — no code changes elsewhere.

### 3.2 Storage
- **ChromaDB** (persistent, local by default) stores embeddings + full text
  + metadata. Cosine similarity space. Chosen for the prototype because it
  requires no external service; swap for a managed store (Pinecone, Azure
  AI Search, pgvector) for production scale/HA by only touching
  `rag/vector_store.py`.

### 3.3 Retrieval & Reranking
1. **Metadata pre-filter**: department/sensitivity constraints applied as a
   `where` clause before vector search — this is the access-control hook
   (see §5).
2. **Dense retrieval**: top-12 candidates via cosine similarity.
3. **Hybrid fusion**: BM25 sparse score blended with dense score (α=0.5),
   which recovers exact-term matches (e.g., "GlobalProtect", policy IDs)
   that embeddings alone can under-rank.
4. **Cross-encoder rerank**: `ms-marco-MiniLM-L-6-v2` jointly scores
   (query, chunk) pairs for the final top-4 — meaningfully more precise
   than embedding similarity alone, at low latency cost since it only runs
   on 12 candidates, not the full corpus.

### 3.4 Generation
- Grounded system prompt forces citation of every claim (`[S1]`, `[S2]`,
  ...), explicit "I don't know" behavior when context is insufficient, and
  conflict surfacing when documents disagree (e.g., stale vs current
  policy version).
- Chat history (last 6 turns) is passed to resolve follow-up references
  ("what about for contractors?").
- **Confidence indicator**: derived from the top reranked relevance score
  — High (≥0.75), Medium (≥0.55), Low (below) — shown in the UI so users
  know when to double check a source themselves.

### 3.5 UI & Evaluation
- **Streamlit chatbot** (`app.py`): chat interface, department/sensitivity
  filters, expandable source citations with page/version/relevance score,
  confidence badge, indexed-document browser.
- **Evaluation dashboard** (`eval/dashboard.py`): retrieval hit-rate, mean
  reciprocal rank, answer keyword coverage, confidence distribution, and
  per-question latency, computed by `eval/evaluate.py` against a labeled
  test set (`eval/eval_testset.json`), including a negative test case that
  checks the system correctly declines to answer when no source exists
  (hallucination guard).

## 4. Why These Design Choices

| Decision | Rationale |
|---|---|
| Hybrid (dense+sparse) retrieval | Pure embeddings miss exact terms (error codes, form names); pure keyword misses paraphrased questions. Combining both is standard enterprise-RAG practice. |
| Cross-encoder rerank | Embedding cosine similarity is a coarse first pass; cross-encoders score query-chunk pairs jointly and materially improve top-k precision, at acceptable latency since it's applied only to ~12 candidates. |
| Metadata-first filtering | Applying department/sensitivity filters *before* semantic search (not after) is both faster and is the natural integration point for real ACLs (Entra ID groups, SharePoint permissions). |
| Chunk-level lineage | Every chunk retains source/page/version, so every answer is traceable — a hard requirement for compliance/HR/legal content. |
| Confidence indicator | Prevents over-trusting low-relevance answers; nudges users to verify when the system itself is unsure. |
| Provider-agnostic LLM client | Enterprises often need to move between OpenAI and Azure OpenAI for data-residency/compliance reasons; isolating this in one file avoids a rewrite. |

## 5. Security & Governance (Prototype → Production Path)
- **Prototype**: metadata-based filtering by department/sensitivity, applied
  at query time via Chroma `where` clauses.
- **Production**: replace with real identity-aware filtering — resolve the
  requesting user's Entra ID/AD groups and SharePoint permissions at query
  time, and only retrieve chunks the user is authorized to see (row-level
  security in the vector store, or a post-retrieval ACL filter before
  chunks ever reach the LLM prompt).
- **Auditability**: log every query, retrieved chunk IDs, and generated
  answer for compliance review (not implemented in the prototype, called
  out here as a deployment requirement).
- **PII/sensitive content**: sensitivity metadata (`Public/Internal/
  Confidential`) should map to real classification labels (e.g., Microsoft
  Purview) rather than the illustrative values used here.

## 6. Evaluation Methodology
- **Retrieval quality**: hit-rate@k and MRR against a labeled test set
  mapping questions to their known source document.
- **Answer quality**: keyword/fact coverage vs. expected answer elements
  (a fast proxy suitable for CI; swap for an LLM-as-judge or RAGAS
  faithfulness/relevancy metrics for deeper production evaluation).
- **Hallucination guard**: a negative test case with no valid source
  verifies the assistant declines rather than fabricating.
- **Operational**: end-to-end latency per question, confidence-label
  distribution across the test set.

## 7. Business Impact
- Reduces time-to-answer from manual document search (often minutes) to
  seconds, with the answer traceable to source for trust and compliance.
- Reduces duplicated effort and inconsistent decisions caused by employees
  referencing outdated document versions (version metadata + conflict
  surfacing addresses this directly).
- Structured for incremental adoption: start with a handful of document
  sets (HR, IT) and expand department coverage without re-architecting.
