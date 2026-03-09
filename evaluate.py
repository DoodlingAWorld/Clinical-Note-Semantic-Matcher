"""
evaluate.py — MRR evaluation for hybrid semantic search.

Runs the golden dataset through Pinecone at multiple alpha values to compare
pure dense (alpha=1.0) against hybrid BM25+dense blends.

MRR = (1/|Q|) * sum(1 / rank_i)
"""

import json
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv

load_dotenv()

TOP_K = 10
INDEX_NAME = "medical-hybrid-search"
MODEL_NAME = "NeuML/pubmedbert-base-embeddings"
GOLDEN_DATASET_FILE = "golden_dataset.json"
BM25_MODEL_FILE = "bm25_model.json"
ALPHAS_TO_TEST = [1.0, 0.75, 0.5, 0.25]

print("Connecting to Pinecone...")
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

print(f"Loading models...")
model = SentenceTransformer(MODEL_NAME)
bm25 = BM25Encoder()
bm25.load(BM25_MODEL_FILE)

print(f"Loading golden dataset...")
with open(GOLDEN_DATASET_FILE, "r", encoding="utf-8") as f:
    test_cases = json.load(f)["queries"]
print(f"Ready. {len(test_cases)} test cases.\n")


def hybrid_scale(dense: list, sparse: dict, alpha: float) -> tuple:
    """Scale dense by alpha and sparse by (1-alpha) before querying.
    Pinecone scores = dot(query_dense, doc_dense) + dot(query_sparse, doc_sparse),
    so scaling the query vectors controls each side's contribution to the final score.
    """
    hsparse = {
        "indices": sparse["indices"],
        "values": [v * (1 - alpha) for v in sparse["values"]],
    }
    return [v * alpha for v in dense], hsparse


def evaluate_mrr(test_cases: list, top_k: int, alpha: float) -> float:
    reciprocal_ranks = []
    hits_at_1 = hits_at_3 = hits_at_5 = 0
    not_found = []

    print(f"{'Query':<60} {'Expected ID':<12} {'Rank':<6} {'RR'}")
    print("-" * 90)

    for test in test_cases:
        query = test["query"]
        expected_id = test["expected_id"]

        hdense, hsparse = hybrid_scale(
            model.encode(query, show_progress_bar=False).tolist(),
            bm25.encode_queries(query),
            alpha,
        )

        results = index.query(
            vector=hdense,
            sparse_vector=hsparse,
            top_k=top_k,
            include_metadata=False,
        )

        rank = next(
            (pos for pos, m in enumerate(results["matches"], 1) if m["id"] == expected_id),
            0,
        )

        if rank > 0:
            rr = 1.0 / rank
            if rank == 1: hits_at_1 += 1
            if rank <= 3: hits_at_3 += 1
            if rank <= 5: hits_at_5 += 1
        else:
            rr = 0.0
            not_found.append({"query": query, "expected_id": expected_id})

        print(f"{query[:58]:<60} {expected_id:<12} {str(rank) if rank else f'>{top_k}':<6} {rr:.3f}")
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
