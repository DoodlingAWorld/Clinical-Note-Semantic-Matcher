"""
generate_golden_dataset.py

Builds the evaluation "answer key" for Phase 1.

The core idea: instead of manually writing queries by hand, we use an LLM to
read each real document and generate a query that only that specific document
could answer. This gives us ground-truth (query -> expected_id) pairs.

The expected_id values match exactly what was upserted to Pinecone in the
notebook: df.index.astype(str) after dropna(). Replicating that preprocessing
here is critical — if even one row was dropped differently, IDs won't align.
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
SAMPLES_PER_CLASS = 10      # How many documents to sample per diagnosis class
TEXT_CHARS_FOR_LLM = 800    # Truncate clinical text sent to LLM (cost control)
OUTPUT_FILE = "golden_dataset.json"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# 1. Load and preprocess — must match the notebook exactly
# ---------------------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv("data/alldata_1_for_kaggle.csv", encoding="latin1")
df = df.rename(columns={"0": "diagnosis", "a": "clinical_text"})
df = df.dropna()
# Pinecone IDs were set as df.index.astype(str) AFTER dropna, so we preserve
# the original index here rather than resetting it.
df["unique_id"] = df.index.astype(str)

print(f"Dataset loaded: {len(df)} rows")
print("\nDiagnosis distribution:")
print(df["diagnosis"].value_counts().to_string())

# ---------------------------------------------------------------------------
# 2. Sample rows proportionally across all classes
# ---------------------------------------------------------------------------
sampled = (
    df.groupby("diagnosis", group_keys=False)
    .apply(lambda x: x.sample(min(len(x), SAMPLES_PER_CLASS), random_state=42))
)
sampled = sampled.reset_index(drop=True)
print(f"\nSampled {len(sampled)} rows across {sampled['diagnosis'].nunique()} classes")

# ---------------------------------------------------------------------------
# 3. Generate queries via LLM
# ---------------------------------------------------------------------------
def generate_query(diagnosis: str, clinical_text: str) -> str:
    """
    Ask gpt-4o-mini to produce one highly specific search query grounded in
    the unique details of this document. The query must be answerable ONLY by
    this document — generic queries like "cancer patient" would match hundreds
    of documents and make the evaluation useless.
    """
    text_excerpt = clinical_text[:TEXT_CHARS_FOR_LLM]

    prompt = f"""You are building an evaluation dataset for a medical semantic search engine.

Given the clinical document below, write ONE specific search query that a clinician or researcher might type to find exactly this document.

Rules:
- Include specific details unique to this document: specific patient demographics, specific procedures, specific findings, specific statistics, or specific institutions mentioned.
- Do NOT start with generic phrases like "patient with cancer" or "study about".
- Do NOT copy the first sentence verbatim. Rephrase.
- The query should be 10-25 words.
- Output ONLY the query text, nothing else.

Diagnosis label: {diagnosis}
Document excerpt:
{text_excerpt}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=80,
    )
    return response.choices[0].message.content.strip()


print("\nGenerating queries via gpt-4o-mini...")
print("(This calls the OpenAI API once per sampled row)\n")

test_cases = []
failed = 0

for i, row in sampled.iterrows():
    try:
        query = generate_query(row["diagnosis"], str(row["clinical_text"]))
        test_cases.append({
            "query": query,
            "expected_id": row["unique_id"],
            "diagnosis": row["diagnosis"],
        })
        print(f"  [{len(test_cases)}/{len(sampled)}] ID {row['unique_id']} ({row['diagnosis']})")
        print(f"    -> {query}\n")
        time.sleep(0.1)

    except Exception as e:
        failed += 1
        print(f"  [FAILED] Row {row['unique_id']}: {e}")

# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------
output = {"queries": test_cases}
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nDone. {len(test_cases)} queries saved to {OUTPUT_FILE} ({failed} failed)")
print("Next step: run evaluate.py to get your baseline MRR score.")
