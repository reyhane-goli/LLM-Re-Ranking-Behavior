try:
    # Newer Pyserini
    from pyserini.search.lucene import LuceneSearcher
except ImportError:
    # Older Pyserini
    from pyserini.search import LuceneSearcher


try:
    from pyserini.search._base import get_topics, get_qrels
except ImportError:
    # for older Pyserini versions
    from pyserini.search import get_topics, get_qrels

from beir import util
# from beir.datasets.data_loader import GenericDataLoader
import os

from tqdm import tqdm

# --- Dataset → index mapping ---


THE_INDEX = {
    'dl19': 'msmarco-v1-passage',
    'dl20': 'msmarco-v1-passage',
   'trec-covid': 'beir-v1.0.0-trec-covid.flat',
    'nfcorpus': 'beir-v1.0.0-nfcorpus.flat',
    'webis-touche2020': 'beir-v1.0.0-webis-touche2020.flat',
    'dbpedia-entity': 'beir-v1.0.0-dbpedia-entity.flat',
    'scifact': 'beir-v1.0.0-scifact.flat',
    'signal1m': 'beir-v1.0.0-signal1m.flat',
    'trec-news': 'beir-v1.0.0-trec-news.flat',
    'robust04': 'beir-v1.0.0-robust04.flat',
    'fiqa': 'beir-v1.0.0-fiqa.flat',
    'scidocs': 'beir-v1.0.0-scidocs.flat',
    'arguana': 'beir-v1.0.0-arguana.flat',
    'quora': 'beir-v1.0.0-quora.flat',
    'fever': 'beir-v1.0.0-fever-flat',
}

THE_TOPICS = {
    'dl19': 'dl19-passage',
    'dl20': 'dl20-passage',
    'trec-covid': 'beir-v1.0.0-trec-covid-test',
    'arguana': 'beir-v1.0.0-arguana-test',
    'webis-touche2020': 'beir-v1.0.0-webis-touche2020-test',
    'trec-news': 'beir-v1.0.0-trec-news-test',
    'scifact': 'beir-v1.0.0-scifact-test',
    'fiqa': 'beir-v1.0.0-fiqa-test',
    'scidocs': 'beir-v1.0.0-scidocs-test',
    'nfcorpus': 'beir-v1.0.0-nfcorpus-test',
    'quora': 'beir-v1.0.0-quora-test',
    'dbpedia-entity': 'beir-v1.0.0-dbpedia-entity-test',
    'fever': 'beir-v1.0.0-fever-test',
    'robust04': 'beir-v1.0.0-robust04-test',
    'signal1m': 'beir-v1.0.0-signal1m-test'

}


def _load_beir_queries_qrels(dataset: str, out_dir: str = "./datasets"):
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    data_path = util.download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    return queries, qrels

def retrieve_topk(dataset, k=100, index_dir=None, **kwargs):
    print(f"=== {dataset} :: BM25 top-{k} ===")

    # --- Case 1: TREC DL19/DL20 (MS MARCO Passage) ---
#     if dataset in ["dl19", "dl20"]:
#         searcher = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")
#         # Use Pyserini topic IDs directly (no "-test")
#         try:
#             topics = get_topics(dataset)
#             qrels = get_qrels(dataset)
#         except ValueError:
#             # fallback for MS MARCO-style datasets
#             topics = get_topics(f"{dataset}-passage")
#             qrels = get_qrels(f"{dataset}-passage")


#         bm25_res = {}
#         for qid, query in topics.items():
#             if isinstance(query, dict):
#                 query_text = query.get('title') or query.get('text') or ''
#             else:
#                 query_text = query
                
#             hits = searcher.search(query_text, k)
            
            
#             bm25_res[qid] = [(hit.docid, hit.score) for hit in hits]

# #             import json

# #             bm25_res[qid] = []
# #             for hit in hits:
# #                 raw_json = searcher.doc(hit.docid).raw()
# #                 try:
# #                     text = json.loads(raw_json).get("contents", "")
# #                 except Exception:
# #                     text = raw_json  # fallback if not JSON
# #                 # Clean and truncate for T5 (512 tokens ≈ 4000 chars)
# #                 text = text.replace('\n', ' ').replace('\t', ' ')
# #                 text = ' '.join(text.split())
# #                 text = text[:4000]
# #                 bm25_res[qid].append((hit.docid, text))


#         print(f"[INFO] Retrieved {len(bm25_res)} queries for {dataset}")
#         return bm25_res, qrels, topics, searcher

    # Case 1 - DL19 and DL 20
    if dataset in ["dl19", "dl20"]:
        index_name = THE_INDEX.get(dataset)
        topics_name = THE_TOPICS.get(dataset, dataset)

        if index_name is None:
            raise ValueError(f"[ERROR] Unknown dataset: {dataset}")

        searcher = LuceneSearcher.from_prebuilt_index(index_name)
        topics = get_topics(topics_name)

        try:
            qrels = get_qrels(topics_name)
        except FileNotFoundError:
            print(f"[WARN] No qrels found for {dataset}, trying fallback ...")
            if dataset == "dl20":
                import urllib.request, os
                os.makedirs("./datasets/qrels", exist_ok=True)
                qrels_path = "./datasets/qrels/qrels.dl20.txt"
                if not os.path.exists(qrels_path):
                    urllib.request.urlretrieve(
                        "https://raw.githubusercontent.com/castorini/anserini/master/src/main/resources/topics-and-qrels/qrels.dl20-passage.txt",
                        qrels_path
                    )
                from pyserini.eval import load_qrels
                qrels = load_qrels(qrels_path)
            else:
                raise


        bm25_res = {}
        for qid, query in topics.items():
            qtext = query.get("title") if isinstance(query, dict) else str(query)
            hits = searcher.search(qtext, k)
            bm25_res[qid] = [(hit.docid, hit.score) for hit in hits]

        print(f"[INFO] Retrieved {len(bm25_res)} queries for {dataset}")
        return bm25_res, qrels, topics, searcher

    # --- Case 2: BEIR datasets ---
    else:
        
        if dataset not in THE_INDEX:
            raise ValueError(f"[ERROR] Unknown BEIR dataset: {dataset}")

        index_name = THE_INDEX[dataset]
        topics_name = THE_TOPICS[dataset]

        print(f"[INFO] Using index:  {index_name}")
        print(f"[INFO] Using topics: {topics_name}")

        # --- Step 1: load index ---
        searcher = LuceneSearcher.from_prebuilt_index(index_name)

        # --- Step 2: load topics ---
        try:
            topics = get_topics(topics_name)
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to load topics for {dataset}: {e}")

        # --- Step 3: load qrels ---
        try:
            qrels = get_qrels(topics_name)
        except FileNotFoundError:
            print(f"[WARN] No qrels found for {dataset}. Continuing without ground truth.")
            qrels = {}

            # --- Step 4: retrieve top-k results ---
        # --- Step 4: retrieve top-k results ---
        results = {}
        for qid, query in topics.items():
            # Some BEIR topics are dicts like {"title": "..."} or {"text": "..."}
            if isinstance(query, dict):
                query_text = query.get("title") or query.get("text") or str(list(query.values())[0])
            else:
                query_text = str(query)

            hits = searcher.search(query_text, k=k)
            results[qid] = [(h.docid, h.score) for h in hits]

        print(f"[INFO] Retrieved {len(topics)} queries for {dataset}")
        return results, qrels, topics, searcher


    # --- Case 2: BEIR datasets ---
#     else:
#         from beir.datasets.data_loader import GenericDataLoader


#         DATASET_ALIASES = {
#             "touche": "webis-touche2020",
#             "dbpedia": "dbpedia-entity",
#             "signal": "signal1m",
#             "news": "trec-news",
#         }
#         dataset = DATASET_ALIASES.get(dataset, dataset)

#         # Choose index
# #         THE_INDEX = {
# #             "trec-covid": "beir-v1.0.0-trec-covid.flat",
# #             "nfcorpus": "beir-v1.0.0-nfcorpus.flat",
# #             "webis-touche2020": "beir-v1.0.0-webis-touche2020.flat",
# #             "dbpedia-entity": "beir-v1.0.0-dbpedia-entity.flat",
# #             "scifact": "beir-v1.0.0-scifact.flat",
# #             "signal1m": "beir-v1.0.0-signal1m.flat",
# #             "trec-news": "beir-v1.0.0-trec-news.flat",
# #             "robust04": "beir-v1.0.0-robust04.flat",
# #         }

#         if dataset not in THE_INDEX:
#             raise ValueError(f"Dataset {dataset} not supported for prebuilt BM25.")

#         index_name = THE_INDEX[dataset]
#         searcher = LuceneSearcher.from_prebuilt_index(index_name)

#         # --- Load topics + qrels ---
#         try:

#             topics = get_topics(dataset)
#             qrels = get_qrels(dataset)
            
#         except Exception:
#             # Fall back to BEIR loader (for trec-covid, nfcorpus, scifact, etc.)
#             from beir import util
#             import os
            
            
# #             data_path = util.download_and_unzip(f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip", "./datasets")

#             from beir import util
#             import tarfile
#             from urllib.request import urlretrieve


#             try:
#                 data_path = util.download_and_unzip(
#                     f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip",
#                     "./datasets"
#                 )
#             except Exception as e:
#                 print(f"[WARN] Failed to unzip {dataset}.zip ({e}), trying Hugging Face mirror...")

#                 # Hugging Face mirror (correct paths for Signal1M & TREC-News)
#                 hf_urls = {
#                     "signal1m": "https://huggingface.co/datasets/beir-corpus/signal1m/resolve/main/signal1m.tar.gz",
#                     "trec-news": "https://huggingface.co/datasets/beir-corpus/trec-news/resolve/main/trec-news.tar.gz",
#                 }

#                 dataset_url = hf_urls.get(dataset)
#                 if dataset_url is None:
#                     raise ValueError(f"[ERROR] No valid URL found for dataset: {dataset}")

#                 tar_gz = f"./datasets/{dataset}.tar.gz"
#                 urlretrieve(dataset_url, tar_gz)

#                 with tarfile.open(tar_gz, "r:gz") as tar:
#                     tar.extractall("./datasets")

#                 data_path = f"./datasets/{dataset}"
#                 print(f"[INFO] Extracted {dataset} successfully from Hugging Face mirror → {data_path}")



            
#             corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
#             topics = {qid: {"title": qtext} for qid, qtext in queries.items()}
#             print(f"[INFO] Loaded {len(topics)} BEIR queries for {dataset}")

#         # --- Retrieval loop ---
#         results = {}
#         for qid, q in topics.items():
#             query = q["title"] if isinstance(q, dict) else q
#             hits = searcher.search(query, k)
#             results[qid] = [(hit.docid, hit.score) for hit in hits]

#         return results, qrels, topics, searcher
    
    
    

#     else:

#         DATASET_ALIASES = {
#             "touche": "webis-touche2020",
#             "dbpedia": "dbpedia-entity",
#             "signal": "signal1m",
#             "news": "trec-news",
#         }
#         dataset = DATASET_ALIASES.get(dataset, dataset)

#         THE_INDEX = {
#             "trec-covid": "beir-v1.0.0-trec-covid.flat",
#             "nfcorpus": "beir-v1.0.0-nfcorpus.flat",
#             "webis-touche2020": "beir-v1.0.0-webis-touche2020.flat",
#             "dbpedia-entity": "beir-v1.0.0-dbpedia-entity.flat",
#             "scifact": "beir-v1.0.0-scifact.flat",
#             "signal1m": "beir-v1.0.0-signal1m.flat",
#             "trec-news": "beir-v1.0.0-trec-news.flat",
#             "robust04": "beir-v1.0.0-robust04.flat",
#         }

#         if dataset not in THE_INDEX:
#             raise ValueError(f"Dataset {dataset} not supported for prebuilt BM25.")

#         index_name = THE_INDEX[dataset]
#         searcher = LuceneSearcher.from_prebuilt_index(index_name)
#         print(f"[INFO] Loaded Pyserini prebuilt index for {dataset}: {index_name}")

#         # --- Load BEIR queries + qrels ---
#         url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
#         data_path = util.download_and_unzip(url, "./datasets")
#         corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

#         # --- Retrieve ---
#         results = {}
#         for qid, qtext in tqdm(queries.items(), desc=f"{dataset} retrieval"):
#             hits = searcher.search(qtext, k)
#             results[qid] = [(hit.docid, hit.score) for hit in hits]

#         print(f"[INFO] Retrieved {len(queries)} BEIR queries for {dataset}")
#         return results, qrels, queries, searcher




    raise ValueError(f"Dataset {dataset} not supported for prebuilt BM25.")
