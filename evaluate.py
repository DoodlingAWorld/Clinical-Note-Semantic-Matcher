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

Why MRR and not just accuracy?
    Accuracy only tells you if the correct doc was in the top-k.
    MRR also tells you *how high up* it ranked. A system that always
    puts the right answer at position 1 (MRR=1.0) is better than one
    that puts it at position 5 even if both have 100% top-5 accuracy.

Interpreting your score:
    0.80 - 1.00  World-class. Correct doc almost always at position 1.
    0.50 - 0.79  Solid baseline. Usually in top 2-3.
    < 0.50       Room for improvement — good motivation for Phase 2 (hybrid search).
"""

import json
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TOP_K = 10   # How many results to retrieve per query.
             # Using 10 instead of 5 so the correct doc has more chances to
             # appear — this gives a fairer picture of the engine's capability.

INDEX_NAME = "medical-weighted-search"
MODEL_NAME = "NeuML/pubmedbert-base-embeddings"
GOLDEN_DATASET_FILE = "golden_dataset.json"

# ---------------------------------------------------------------------------
# 1. Connect to Pinecone
# ---------------------------------------------------------------------------
print("Connecting to Pinecone...")
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

# ---------------------------------------------------------------------------
# 2. Load embedding model
# ---------------------------------------------------------------------------
print(f"Loading embedding model: {MODEL_NAME}")
print("(Uses the same model as the notebook — critical for consistent vector space)\n")
model = SentenceTransformer(MODEL_NAME)

# ---------------------------------------------------------------------------
# 3. Load golden dataset
# ---------------------------------------------------------------------------
print(f"Loading golden dataset from {GOLDEN_DATASET_FILE}...")
with open(GOLDEN_DATASET_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

test_cases = data["queries"]
print(f"Loaded {len(test_cases)} test cases\n")

# ---------------------------------------------------------------------------
# 4. Run evaluation
# ---------------------------------------------------------------------------
def evaluate_mrr(test_cases: list, top_k: int = TOP_K) -> float:
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

        # Embed the query using the same model that embedded the documents.
        # Note: we encode as a plain list here — NOT wrapped in another list.
        # Gemini's sample code had vector=[query_embedding] which is a bug
        # (passes a 2D array; Pinecone expects a 1D vector).
        query_embedding = model.encode(query, show_progress_bar=False).tolist()

        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=False,
        )

        # Walk through results to find where the expected doc landed
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
    print(f"RESULTS ({n} queries, top_k={top_k})")
    print("=" * 90)
    print(f"  MRR Score:       {mrr:.4f}")
    print(f"  Hit@1:           {hits_at_1}/{n}  ({100*hits_at_1/n:.1f}%)")
    print(f"  Hit@3:           {hits_at_3}/{n}  ({100*hits_at_3/n:.1f}%)")
    print(f"  Hit@5:           {hits_at_5}/{n}  ({100*hits_at_5/n:.1f}%)")
    print(f"  Not in top {top_k}:   {len(not_found)}/{n}")

    if not_found:
        print("\nFailed queries (correct doc not found in top results):")
        for item in not_found:
            print(f"  ID {item['expected_id']}: {item['query'][:80]}")

    return mrr


mrr_score = evaluate_mrr(test_cases, top_k=TOP_K)
print(f"\nBaseline MRR = {mrr_score:.4f}")
print("Save this number — it is your benchmark to beat after adding hybrid search.")
