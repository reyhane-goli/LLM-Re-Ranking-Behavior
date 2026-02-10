from trec_eval import EvalFunction
import json
import os

save_dir = 'saved_results'
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

THE_INDEX = {
    'dl19': 'msmarco-v1-passage',
    'dl20': 'msmarco-v1-passage',
    'covid': 'beir-v1.0.0-trec-covid.flat',
    'arguana': 'beir-v1.0.0-arguana.flat',
    'touche': 'beir-v1.0.0-webis-touche2020.flat',
    'news': 'beir-v1.0.0-trec-news.flat',
    'scifact': 'beir-v1.0.0-scifact.flat',
    'fiqa': 'beir-v1.0.0-fiqa.flat',
    'scidocs': 'beir-v1.0.0-scidocs.flat',
    'nfc': 'beir-v1.0.0-nfcorpus.flat',
    'quora': 'beir-v1.0.0-quora.flat',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity.flat',
    'fever': 'beir-v1.0.0-fever-flat',
    'robust04': 'beir-v1.0.0-robust04.flat',
    'signal': 'beir-v1.0.0-signal1m.flat',

    'mrtydi-ar': 'mrtydi-v1.1-arabic',
    'mrtydi-bn': 'mrtydi-v1.1-bengali',
    'mrtydi-fi': 'mrtydi-v1.1-finnish',
    'mrtydi-id': 'mrtydi-v1.1-indonesian',
    'mrtydi-ja': 'mrtydi-v1.1-japanese',
    'mrtydi-ko': 'mrtydi-v1.1-korean',
    'mrtydi-ru': 'mrtydi-v1.1-russian',
    'mrtydi-sw': 'mrtydi-v1.1-swahili',
    'mrtydi-te': 'mrtydi-v1.1-telugu',
    'mrtydi-th': 'mrtydi-v1.1-thai',
}

THE_TOPICS = {
    'dl19': 'dl19-passage',
    'dl20': 'dl20-passage',
    'covid': 'beir-v1.0.0-trec-covid-test',
    'arguana': 'beir-v1.0.0-arguana-test',
    'touche': 'beir-v1.0.0-webis-touche2020-test',
    'news': 'beir-v1.0.0-trec-news-test',
    'scifact': 'beir-v1.0.0-scifact-test',
    'fiqa': 'beir-v1.0.0-fiqa-test',
    'scidocs': 'beir-v1.0.0-scidocs-test',
    'nfc': 'beir-v1.0.0-nfcorpus-test',
    'quora': 'beir-v1.0.0-quora-test',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity-test',
    'fever': 'beir-v1.0.0-fever-test',
    'robust04': 'beir-v1.0.0-robust04-test',
    'signal': 'beir-v1.0.0-signal1m-test',

    'mrtydi-ar': 'mrtydi-v1.1-arabic-test',
    'mrtydi-bn': 'mrtydi-v1.1-bengali-test',
    'mrtydi-fi': 'mrtydi-v1.1-finnish-test',
    'mrtydi-id': 'mrtydi-v1.1-indonesian-test',
    'mrtydi-ja': 'mrtydi-v1.1-japanese-test',
    'mrtydi-ko': 'mrtydi-v1.1-korean-test',
    'mrtydi-ru': 'mrtydi-v1.1-russian-test',
    'mrtydi-sw': 'mrtydi-v1.1-swahili-test',
    'mrtydi-te': 'mrtydi-v1.1-telugu-test',
    'mrtydi-th': 'mrtydi-v1.1-thai-test',

}

from rank_gpt import run_retriever, sliding_windows, write_eval_file
from pyserini.search import get_topics, get_qrels
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
import tempfile
import os
import json
import shutil

openai_key = ""

# # for data in ['dl19', 'dl20', 'covid', 'nfc', 'touche', 'dbpedia', 'scifact', 'signal', 'news', 'robust04']:
for data in ['covid']:
    print('#' * 20)
    print(f'Evaluation on {data}')
    print('#' * 20)

    
    # Retrieve passages using pyserini BM25.
    try:
        searcher = LuceneSearcher.from_prebuilt_index(THE_INDEX[data])
        topics = get_topics(THE_TOPICS[data] if data != 'dl20' else 'dl20')
        qrels = get_qrels(THE_TOPICS[data])
        rank_results = run_retriever(topics, searcher, qrels, k=100)
    except:
        print(f'Failed to retrieve passages for {data}')
        continue
    
    # Run sliding window permutation generation
    new_results = []
    for item in tqdm(rank_results):
        new_item = sliding_windows(item, rank_start=0, rank_end=99, window_size=20, step=10,
                                   model_name='gpt-3.5-turbo-0125', api_key=openai_key)
        new_results.append(new_item)

    # Evaluate nDCG@10
    
    save_path = os.path.join(save_dir, f'{data}_gpt_0125_reranked_2025_05_28_99Item.json')
#     save_path = os.path.join(save_dir, f'{data}_gpt_reranked_1106.json')
# save_path = os.path.join(save_dir, 'covid_gpt_reranked.json')


    # Save new_results to a JSON file
    with open(save_path, 'w') as f:
        json.dump(new_results, f)

    print(f'Saved reranked results to {save_path}')
    
    
# Load reranked results from file
# with open(save_path, 'r') as f:
#     new_results = json.load(f)


    # Create an empty text file to write results, and pass the name to eval
    temp_file = tempfile.NamedTemporaryFile(delete=False).name
    EvalFunction.write_file(new_results, temp_file)
    EvalFunction.main(THE_TOPICS[data], temp_file)
#     EvalFunction.main(THE_TOPICS['covid'], temp_file)


# for data in ['mrtydi-ar', 'mrtydi-bn', 'mrtydi-fi', 'mrtydi-id', 'mrtydi-ja', 'mrtydi-ko', 'mrtydi-ru', 'mrtydi-sw', 'mrtydi-te', 'mrtydi-th']:
#     print('#' * 20)
#     print(f'Evaluation on {data}')
#     print('#' * 20)

#     # Retrieve passages using pyserini BM25.
#     try:
#         searcher = LuceneSearcher.from_prebuilt_index(THE_INDEX[data])
#         topics = get_topics(THE_TOPICS[data] if data != 'dl20' else 'dl20')
#         qrels = get_qrels(THE_TOPICS[data])
#         rank_results = run_retriever(topics, searcher, qrels, k=100)
#         rank_results = rank_results[:100]

#     except:
#         print(f'Failed to retrieve passages for {data}')
#         continue

#     # Run sliding window permutation generation
#     new_results = []
#     for item in tqdm(rank_results):
#         new_item = sliding_windows(item, rank_start=0, rank_end=100, window_size=20, step=10,
#                                    model_name='gpt-3.5-turbo', api_key=openai_key)
#         new_results.append(new_item)

#     # Evaluate nDCG@10
#     from trec_eval import EvalFunction

#     temp_file = tempfile.NamedTemporaryFile(delete=False).name
#     EvalFunction.write_file(new_results, temp_file)
#     EvalFunction.main(THE_TOPICS[data], temp_file)
