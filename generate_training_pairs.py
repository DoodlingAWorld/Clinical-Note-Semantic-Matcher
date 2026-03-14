"""
generate_training_pairs.py

Generates 200 (query, document) training pairs for Phase 4 contrastive fine-tuning.

Two key differences from generate_golden_dataset.py:
  1. Excludes rows already used in golden_dataset.json — no data leakage between
     the training set and the evaluation set.
  2. Saves the document text alongside the query, because the fine-tuning notebook
     needs both sides of each pair as raw text.

Output format — training_pairs.json:
  {
    "pairs": [
      { "query": "...", "document": "Thyroid_Cancer. <text excerpt>", "id": "42" },
      ...
    ]
  }

Why this format?
  sentence-transformers' MultipleNegativesRankingLoss expects InputExample(texts=[anchor, positive]).
  The anchor is the query. The positive is the document text.
  We prepend the diagnosis label to the document so the model sees it as part of the text,
  mirroring the spirit of the weighted embeddings used at inference time.
"""

import json
import os
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLES_PER_CLASS = 40      # 40 per class × 5 classes = 200 pairs total
TEXT_CHARS_FOR_LLM = 800    # How much text to show the LLM when generating queries
DOCUMENT_CHARS = 500        # How much text to store as the "positive" document
OUTPUT_FILE = "training_pairs.json"
EVAL_FILE = "golden_dataset.json"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# 1. Load dataset — same preprocessing as the notebook
# ---------------------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv("data/alldata_1_for_kaggle.csv", encoding="latin1")
df = df.rename(columns={"0": "diagnosis", "a": "clinical_text"})
df = df.dropna()
df["unique_id"] = df.index.astype(str)

print(f"Dataset: {len(df)} rows")
print("\nClass distribution:")
print(df["diagnosis"].value_counts().to_string())

# ---------------------------------------------------------------------------
# 2. Exclude IDs already in the evaluation set
#    This is the most important step — training on eval queries would inflate
#    MRR artificially and make the fine-tuning results meaningless.
# ---------------------------------------------------------------------------
with open(EVAL_FILE, "r") as f:
    eval_ids = {q["expected_id"] for q in json.load(f)["queries"]}

df_available = df[~df["unique_id"].isin(eval_ids)]
print(f"\nRows available after excluding {len(eval_ids)} eval IDs: {len(df_available)}")

# ---------------------------------------------------------------------------
# 3. Sample proportionally across classes
# ---------------------------------------------------------------------------
sampled = (
    df_available.groupby("diagnosis", group_keys=False)
    .apply(lambda x: x.sample(min(len(x), SAMPLES_PER_CLASS), random_state=99))
)
sampled = sampled.reset_index(drop=True)
print(f"\nSampled {len(sampled)} rows across {sampled['diagnosis'].nunique()} classes")
print(sampled["diagnosis"].value_counts().to_string())

# ---------------------------------------------------------------------------
# 4. Generate queries via LLM — same prompt logic as golden dataset
# ---------------------------------------------------------------------------
def generate_query(diagnosis: str, clinical_text: str) -> str:
    """Generate a specific query that only this document could answer."""
    prompt = f"""You are building a training dataset for fine-tuning a medical search model.

Given the clinical document below, write ONE specific search query a clinician might type to find this exact document.

Rules:
- Include specific details unique to this document: patient demographics, procedures, findings, statistics, or institutions.
- Do NOT use generic phrases like "patient with cancer" or "study about".
- Do NOT copy the first sentence verbatim.
- The query should be 10-25 words.
- Output ONLY the query text, nothing else.

Diagnosis: {diagnosis}
Document:
{clinical_text[:TEXT_CHARS_FOR_LLM]}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=80,
    )
    return response.choices[0].message.content.strip()


print(f"\nGenerating {len(sampled)} queries via gpt-4o-mini...\n")

pairs = []
failed = 0

for _, row in sampled.iterrows():
    try:
        query = generate_query(row["diagnosis"], str(row["clinical_text"]))

        # The "document" is what the model learns to associate with this query.
        # We prepend the diagnosis label so the positive text includes both the
        # class signal and the content — mirrors the weighted embedding approach.
        document = f"{row['diagnosis']}. {str(row['clinical_text'])[:DOCUMENT_CHARS]}"

        pairs.append({
            "query": query,
            "document": document,
            "id": row["unique_id"],
            "diagnosis": row["diagnosis"],
        })

        print(f"  [{len(pairs)}/{len(sampled)}] ID {row['unique_id']} ({row['diagnosis']})")
        print(f"    -> {query}\n")
        time.sleep(0.1)

    except Exception as e:
        failed += 1
        print(f"  [FAILED] Row {row['unique_id']}: {e}")

# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------
output = {"pairs": pairs}
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nDone. {len(pairs)} pairs saved to {OUTPUT_FILE} ({failed} failed)")
print("\nClass breakdown in training set:")
from collections import Counter
counts = Counter(p["diagnosis"] for p in pairs)
for diagnosis, count in sorted(counts.items()):
    print(f"  {diagnosis}: {count}")

print(f"\nNext step: open Colab, upload training_pairs.json, run the fine-tuning notebook.")
