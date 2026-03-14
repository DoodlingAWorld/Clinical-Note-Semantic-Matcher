import json, pandas as pd

df = pd.read_csv('data/alldata_1_for_kaggle.csv', encoding='latin1')
df = df.rename(columns={'0': 'diagnosis', 'a': 'clinical_text'}).dropna()
df['unique_id'] = df.index.astype(str)

with open('golden_dataset.json', encoding='utf-8') as f:
    eval_ids = {q['expected_id'] for q in json.load(f)['queries']}
with open('training_pairs.json', encoding='utf-8') as f:
    train_ids = {p['id'] for p in json.load(f)['pairs']}

used = eval_ids | train_ids
available = df[~df['unique_id'].isin(used)]
print('Available rows per class (not yet used):')
print(available['diagnosis'].value_counts().to_string())
print(f'\nTotal available: {len(available)}')
