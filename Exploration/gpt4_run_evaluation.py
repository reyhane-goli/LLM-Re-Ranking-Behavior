import os
import json
from tqdm import tqdm
import tempfile
from rank_gpt import sliding_windows 
from trec_eval import EvalFunction
from pyserini.search import get_topics, get_qrels
from pyserini.search.lucene import LuceneSearcher



save_dir = 'saved_results'
os.makedirs(save_dir, exist_ok=True)


data = 'covid'

# Load previous GPT-3.5 reranked results
reranked_file = os.path.join(save_dir, f'{data}_gpt_reranked.json')

with open(reranked_file, 'r') as f:
    gpt35_results = json.load(f)

# Prepare new result list
final_results = []

# Rerank top 30 for each query using GPT-4-0314
for item in tqdm(gpt35_results, desc="Reranking top-30 with GPT-4"):
    hits = item['hits']

    top30 = hits[:30]
    bottom70 = hits[30:]

    temp_item = {
        'hits': top30,
        'query': item['query']  # only 'query' is needed
    }

    # Re-rank top30 using GPT-4
    new_top30_item = sliding_windows(temp_item, rank_start=0, rank_end=30, window_size=20, step=10,
                                     model_name='gpt-4-0613', api_key=openai_key)

    # Merge reranked top30 with original bottom70
    new_hits = new_top30_item['hits'] + bottom70

    # Save back into structure
    new_item = {
        'query': item['query'],
        'hits': new_hits
    }
    final_results.append(new_item)

# Save final reranked results
new_save_path = os.path.join(save_dir, f'{data}_gpt4top30_reranked_0613.json')

with open(new_save_path, 'w') as f:
    json.dump(final_results, f)

print(f'Final reranked results saved to: {new_save_path}')

# (Optional) Evaluate the new results
temp_file = tempfile.NamedTemporaryFile(delete=False).name
EvalFunction.write_file(final_results, temp_file)
EvalFunction.main('beir-v1.0.0-trec-covid-test', temp_file)  # Correct name for COVID qrels
