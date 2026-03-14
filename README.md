# Clinical Note Semantic Matcher

![img.png](images/img.png)

A full-stack medical AI search engine built on PubMedBERT, Pinecone, and GPT-4o-mini. The project was built in phases, each adding a measurable layer of ML complexity on top of the previous one — starting from basic vector search and ending at a grounded, citation-backed RAG pipeline.

The dataset is a Kaggle corpus of 7,570 clinical research papers across multiple cancer types (Thyroid, Colon, Lung, etc.).

---

## Architecture

```
User Query
    ↓
Hybrid Retrieval  →  PubMedBERT (dense, semantic meaning)
                  +  BM25 (sparse, exact keyword matching)
                  →  alpha slider blends both signals
    ↓
Top 5 Documents from Pinecone (dotproduct index)
    ↓
RAG Synthesis  →  GPT-4o-mini constrained to retrieved context only
    ↓
Synthesized answer with [Doc N] citations + source cards
```

**Why weighted embeddings?** Standard `model.encode(full_text)` on a 3000-word clinical paper produces a diluted vector — the diagnosis label gets drowned out by procedural language. The pipeline encodes diagnosis (×5), abstract (×3), and full text (×1) separately and combines them, so the final vector sits closer to the diagnosis cluster in embedding space.

**Why hybrid search?** Dense vectors semantically blur specific identifiers — `NAP1L1`, `Ion Torrent`, `IL-1` are treated as "just another medical term." BM25 scores exact token matches with high IDF weight for rare terms, recovering failures that pure semantic search misses.

**Why RAG over plain search?** Returning a ranked list puts the synthesis burden on the user. The RAG layer uses GPT-4o-mini constrained to only the retrieved documents — no outside knowledge, no hallucinations, every claim tied to a specific source.

---

## Evaluation Results (MRR @ top-10)

| Phase | Method | MRR |
|---|---|---|
| Phase 1 | Dense only — PubMedBERT + cosine index | 0.3032 |
| Phase 2 | Hybrid — BM25 + PubMedBERT, alpha=0.75 | 0.3206 |

MRR (Mean Reciprocal Rank) measures the average reciprocal position of the correct document across 50 test queries. Score of 1.0 means the correct document is always returned at position 1.

The evaluation dataset was auto-generated using `gpt-4o-mini`: each test query was grounded in unique, document-specific details (specific gene names, patient demographics, statistics) to prevent generic queries that would match many documents equally.

**Key finding:** Pure keyword search (alpha=0.0) scored 0.2999 — worse than pure semantic (0.3014). Medical literature rewards semantic understanding. However, a small BM25 contribution (alpha=0.75) recovers edge cases involving exact technical identifiers without degrading the semantic queries.

---

## Setup

```bash
pip install streamlit sentence-transformers pinecone pinecone-text python-dotenv pandas openai
```

Create a `.env` file in the project root:
```
PINECONE_API_KEY=your_pinecone_key
OPENAI_API_KEY=your_openai_key
```

Run the notebook (`Cancer Doc Classification.ipynb`) top to bottom to embed and index the dataset, then:

```bash
streamlit run app.py
```

To re-run evaluation:
```bash
python evaluate.py
```

---

## Project Structure

```
├── Cancer Doc Classification.ipynb  # Embedding pipeline + Pinecone indexing
├── app.py                           # Streamlit UI with hybrid search + RAG
├── evaluate.py                      # MRR evaluation across alpha values
├── generate_golden_dataset.py       # Auto-generates evaluation query set via LLM
├── golden_dataset.json              # 50 ground-truth query→document pairs
├── bm25_model.json                  # Fitted BM25 encoder (saved after corpus fit)
└── data/
    └── alldata_1_for_kaggle.csv     # 7,570 clinical research papers
```

---

## Roadmap

- [x] Phase 1 — Weighted PubMedBERT embeddings + cosine vector search
- [x] Phase 1 — MRR evaluation pipeline with auto-generated golden dataset
- [x] Phase 2 — Hybrid BM25 + dense search with alpha blending
- [x] Phase 3 — RAG synthesis layer with GPT-4o-mini and source citations
- [ ] Phase 4 — Contrastive fine-tuning of PubMedBERT on this dataset
