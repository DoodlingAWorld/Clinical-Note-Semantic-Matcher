# Clinical Note Semantic Matcher

![img.png](images/img.png)

A medical AI search engine built on PubMedBERT, Pinecone, and GPT-4o-mini. I built this in phases, with each phase adding a measurable layer of ML complexity on top of the previous one. It starts from basic vector search and ends at a contrastive fine-tuned retrieval pipeline backed by a RAG layer.

The dataset is a Kaggle corpus of 7,570 clinical research papers across three cancer types (Thyroid, Colon, Lung).

---

## Architecture

```
User Query
    |
Hybrid Retrieval  ->  Fine-tuned PubMedBERT (dense, semantic meaning)
                  +   BM25 (sparse, exact keyword matching, alpha-blended)
    |
Top 5 Documents from Pinecone (dotproduct index)
    |
RAG Synthesis  ->  GPT-4o-mini constrained to retrieved context only
    |
Synthesized answer with [Doc N] citations + source cards
```

**Why weighted embeddings?** Running `model.encode(full_text)` on a 3000-word clinical paper produces a diluted vector because the diagnosis label gets drowned out by procedural language. Instead, the pipeline encodes the diagnosis (x5), abstract (x3), and full text (x1) separately and combines them. This pulls the final vector closer to the correct diagnosis cluster in embedding space.

**Why hybrid search?** Dense vectors tend to blur specific technical identifiers. Terms like `NAP1L1`, `Ion Torrent`, and `IL-1` get treated as generic medical terms rather than precise identifiers. BM25 fixes this by scoring exact token matches with high IDF weight for rare terms, recovering the cases that pure semantic search misses.

**Why RAG over plain search?** A ranked list of documents puts all the synthesis work on the user. The RAG layer feeds the retrieved documents into GPT-4o-mini with a strict prompt that limits it to only the provided context, so every claim in the answer is tied to a specific retrieved source.

**Why fine-tune?** Off-the-shelf PubMedBERT understands general medical language but has not been trained for retrieval on this specific corpus. Contrastive fine-tuning with `MultipleNegativesRankingLoss` directly optimizes the embedding space so that query vectors land close to their matching documents.

---

## Evaluation Results (MRR @ top-10)

| Phase | Method | MRR |
|---|---|---|
| Phase 1 | Dense only, PubMedBERT + cosine index | 0.3032 |
| Phase 2 | Hybrid, BM25 + PubMedBERT at alpha=0.75 | 0.3206 |
| Phase 4 | Fine-tuned PubMedBERT, pure dense (alpha=1.0) | **0.3405** |

MRR (Mean Reciprocal Rank) measures the average reciprocal position of the correct document across 50 test queries. A score of 1.0 means the correct document is always the top result.

The evaluation set was auto-generated using `gpt-4o-mini`. Each query was grounded in unique, document-specific details like gene names, patient demographics, and statistics, so queries pinpoint a single document rather than matching many equally.

**Phase 2 finding:** Pure keyword search (alpha=0.0) scored 0.2999, which is actually worse than pure semantic search (0.3014). Medical literature rewards semantic understanding. Adding a small BM25 contribution at alpha=0.75 recovered the failures on exact technical identifiers without hurting the semantic queries.

**Phase 4 finding:** After fine-tuning, pure dense (alpha=1.0) became the best configuration and hybrid search started hurting performance. The fine-tuned model learned to handle the exact-term edge cases that BM25 was patching in Phase 2. Once the retriever is strong enough, the keyword fallback just adds noise.

---

## Fine-Tuning Details

- **Method:** Contrastive learning with `MultipleNegativesRankingLoss`
- **Training pairs:** 600 (query, document) pairs auto-generated via `gpt-4o-mini`, sampled proportionally across all three cancer classes
- **Evaluation set:** 50 held-out pairs, never seen during training
- **Base model:** `NeuML/pubmedbert-base-embeddings` (110M parameters)
- **Training:** 4 epochs, batch size 16, 152 steps on Google Colab T4 GPU (about 75 seconds)
- **In-batch negatives per step:** 16 x 15 = 240

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
├── generate_training_pairs.py       # Auto-generates contrastive training pairs via LLM
├── golden_dataset.json              # 50 held-out ground-truth query-to-document pairs
├── training_pairs.json              # 600 training pairs for fine-tuning
├── bm25_model.json                  # Fitted BM25 encoder
├── finetuned-pubmedbert/            # Fine-tuned model weights
└── data/
    └── alldata_1_for_kaggle.csv     # 7,570 clinical research papers
```

---

## Roadmap

- [x] Phase 1 - Weighted PubMedBERT embeddings + cosine vector search
- [x] Phase 1 - MRR evaluation pipeline with auto-generated golden dataset
- [x] Phase 2 - Hybrid BM25 + dense search with alpha blending
- [x] Phase 3 - RAG synthesis layer with GPT-4o-mini and source citations
- [x] Phase 4 - Contrastive fine-tuning of PubMedBERT, MRR 0.3032 to 0.3405
