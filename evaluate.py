"""
evaluate.py

Measures the quality of your Pinecone semantic search engine using
Mean Reciprocal Rank (MRR).

MRR formula:
    MRR = (1 / |Q|) * sum(1 / rank_i)

Where:
    |Q|    = total number of test queries
    rank_i = the position of the correct document in the result list
             (1-indexed: position 1 is the top result)

If the correct document appears at position 1 -> score = 1.0
If it appears at position 2 -> score = 0.5
If it appears at position 3 -> score = 0.33
If it does not appear in top_k -> score = 0.0

New addition: runs at multiple alpha values so you can see exactly
where hybrid search improves over pure dense retrieval.

    alpha = 1.0  ->  pure dense (semantic only)   <- Previous baseline above
    alpha = 0.75 ->  mostly semantic, some keyword
    alpha = 0.5  ->  equal blend
    alpha = 0.25 ->  mostly keyword
"""

import json
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TOP_K = 10
INDEX_NAME = "medical-hybrid-search"
MODEL_NAME = "NeuML/pubmedbert-base-embeddings"
GOLDEN_DATASET_FILE = "golden_dataset.json"
BM25_MODEL_FILE = "bm25_model.json"

# Running all alphas in one pass shows us where hybrid blending helps.
ALPHAS_TO_TEST = [1.0, 0.75, 0.5, 0.25]

# ---------------------------------------------------------------------------
# 1. Connect to Pinecone
# ---------------------------------------------------------------------------
print("Connecting to Pinecone...")
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

# ---------------------------------------------------------------------------
# 2. Load dense embedding model
# ---------------------------------------------------------------------------
print(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# ---------------------------------------------------------------------------
# 3. Load BM25 model
# ---------------------------------------------------------------------------
print(f"Loading BM25 model from {BM25_MODEL_FILE}...")
bm25 = BM25Encoder()
bm25.load(BM25_MODEL_FILE)
print("Models ready.\n")

# ---------------------------------------------------------------------------
# 4. Load golden dataset
# ---------------------------------------------------------------------------
print(f"Loading golden dataset from {GOLDEN_DATASET_FILE}...")
with open(GOLDEN_DATASET_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

test_cases = data["queries"]
print(f"Loaded {len(test_cases)} test cases\n")

# ---------------------------------------------------------------------------
# 5. Hybrid scaling
# ---------------------------------------------------------------------------
def hybrid_scale(dense: list, sparse: dict, alpha: float) -> tuple:
    """
    Scale dense and sparse vectors by alpha before sending to Pinecone.

    Why we do this client-side: Pinecone uses dotproduct scoring, so
    the final score for a document is:
        dot(query_dense, doc_dense) + dot(query_sparse, doc_sparse)

    By scaling query_dense by alpha and query_sparse by (1-alpha),
    we control how much each side contributes to that final score.

    alpha=1.0 -> only dense contributes  (pure semantic)
    alpha=0.0 -> only sparse contributes (pure keyword)
    alpha=0.5 -> equal contribution
    """
    hsparse = {
        "indices": sparse["indices"],
        "values": [v * (1 - alpha) for v in sparse["values"]],
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse


# 6. Evaluation loop
def evaluate_mrr(test_cases: list, top_k: int, alpha: float) -> float:
    reciprocal_ranks = []
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    not_found = []

    print(f"{'Query':<60} {'Expected ID':<12} {'Rank':<6} {'RR'}")
    print("-" * 90)

    for test in test_cases:
        query = test["query"]
        expected_id = test["expected_id"]

        # Encode with both models
        dense_vec = model.encode(query, show_progress_bar=False).tolist()
        sparse_vec = bm25.encode_queries(query)

        # Scale by alpha
        hdense, hsparse = hybrid_scale(dense_vec, sparse_vec, alpha)

        results = index.query(
            vector=hdense,
            sparse_vector=hsparse,
            top_k=top_k,
            include_metadata=False,
        )

        rank = 0
        for position, match in enumerate(results["matches"], start=1):
            if match["id"] == expected_id:
                rank = position
                break

        if rank > 0:
            rr = 1.0 / rank
            if rank == 1: hits_at_1 += 1
            if rank <= 3: hits_at_3 += 1
            if rank <= 5: hits_at_5 += 1
        else:
            rr = 0.0
            not_found.append({"query": query, "expected_id": expected_id})

        rank_display = str(rank) if rank > 0 else f">{top_k}"
        print(f"{query[:58]:<60} {expected_id:<12} {rank_display:<6} {rr:.3f}")
        reciprocal_ranks.append(rr)

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    n = len(test_cases)

    print("\n" + "=" * 90)
    print(f"RESULTS  alpha={alpha}  ({n} queries, top_k={top_k})")
    print("=" * 90)
    print(f"  MRR Score:       {mrr:.4f}")
    print(f"  Hit@1:           {hits_at_1}/{n}  ({100*hits_at_1/n:.1f}%)")
    print(f"  Hit@3:           {hits_at_3}/{n}  ({100*hits_at_3/n:.1f}%)")
    print(f"  Hit@5:           {hits_at_5}/{n}  ({100*hits_at_5/n:.1f}%)")
    print(f"  Not in top {top_k}:   {len(not_found)}/{n}\n")

    return mrr

# 7. Run all alphas and print comparison
results = {}
for alpha in ALPHAS_TO_TEST:
    print(f"\n{'='*90}")
    print(f"  RUNNING alpha={alpha}")
    print(f"{'='*90}\n")
    results[alpha] = evaluate_mrr(test_cases, top_k=TOP_K, alpha=alpha)

print("\n" + "=" * 50)
print("ALPHA COMPARISON SUMMARY")
print("=" * 50)
print(f"  {'Alpha':<10} {'MRR':<10} {'vs baseline'}")
print(f"  {'-'*35}")
baseline = results[1.0]
for alpha, mrr in results.items():
    diff = mrr - baseline
    diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
    marker = " <-- best" if mrr == max(results.values()) else ""
    print(f"  {alpha:<10} {mrr:<10.4f} {diff_str}{marker}")

print(f"\nPhase 1 baseline (cosine index): 0.3032")
print(f"Phase 2 best (hybrid index):     {max(results.values()):.4f}")
