# import os, json, re, time
# from typing import Dict, List, Tuple, Callable
# from tqdm import tqdm


# OPENAI_API_KEY= ""

# COHERE_API_KEY= ""


# def _have_openai():
#     return os.environ.get("OPENAI_API_KEY") is not None

# def _have_cohere():
#     return os.environ.get("COHERE_API_KEY") is not None

# def _openai_rank(query: str, docids: List[str], passages: List[str], model: str):
#     from openai import OpenAI
#     client = OpenAI()
#     items = [{"id": d, "text": p} for d,p in zip(docids, passages)]
#     prompt = f"Query: {query}\nPassages:\n" + "\n".join([f"- ({it['id']}) {it['text']}" for it in items]) + "\nReturn a JSON array of ids."
#     msg = client.chat.completions.create(model=model, messages=[{"role":"system","content":"You are an expert ranker."},{"role":"user","content":prompt}], temperature=0)
#     text = msg.choices[0].message.content
#     try:
#         ids = json.loads(text)
#     except Exception:
#         ids = [m.strip() for m in re.findall(r"\(([^)]+)\)", text)]
#     order = {docid:i for i,docid in enumerate(ids)}
#     scores = [1.0/(1+order.get(d, 9999)) for d in docids]
#     return scores

# def _cohere_rank(query: str, docids: List[str], passages: List[str], model: str):
#     import cohere
#     co = cohere.Client(os.environ.get("COHERE_API_KEY"))
#     out = co.rerank(model=model, query=query, documents=passages, top_n=len(passages))
#     index_to_score = {r.index: float(len(passages)-i) for i,r in enumerate(out.results)}
#     scores = [index_to_score.get(i, 0.0) for i in range(len(passages))]
#     return scores

# def rerank_with_progress(kind: str, bm25_results: Dict[str, List[Tuple[str, float]]], queries: Dict[str, str], doc_fetcher: Callable[[str], str], batch_size: int = 8, openai_model: str = "gpt-3.5-turbo", colour: str = "green") -> Dict[str, List[Tuple[str, float]]]:
#     if kind in ("gpt35","gpt4") and not _have_openai():
#         print(f"[WARN] Skipping {kind}: OPENAI_API_KEY not set.", flush=True)
#         return {qid: items for qid, items in bm25_results.items()}
#     if kind == "cohere" and not _have_cohere():
#         print(f"[WARN] Skipping Cohere: COHERE_API_KEY not set.", flush=True)
#         return {qid: items for qid, items in bm25_results.items()}

#     name = {"gpt35":"gpt-3.5","gpt4":"gpt-4","cohere":"Cohere"}[kind]
#     print(f"[INFO] Starting {name} reranking via API ({len(bm25_results)} queries × {len(next(iter(bm25_results.values()))) if bm25_results else 0} docs)", flush=True)
#     start = time.time()
#     run = {}
#     iterator = tqdm(bm25_results.items(), total=len(bm25_results), desc=f"[{name}] rerank", ncols=100, colour=colour)
#     for qid, items in iterator:
#         q = queries[qid]
#         docids = [docid for docid,_ in items]
#         passages = [doc_fetcher(did) for did in docids]
#         if kind == "gpt35":
#             scores = _openai_rank(q, docids, passages, openai_model)
#         elif kind == "gpt4":
#             scores = _openai_rank(q, docids, passages, openai_model)
#         elif kind == "cohere":
#             scores = _cohere_rank(q, docids, passages, model="rerank-english-v3.0")
#         else:
#             raise ValueError("Unknown LLM reranker type")
#         paired = list(zip(docids, scores))
#         paired.sort(key=lambda x: x[1], reverse=True)
#         run[qid] = paired
#     dur = time.time()-start
#     print(f"[INFO] {name} reranking complete in {int(dur//60)}m{int(dur%60):02d}s.", flush=True)
#     return run




# ===================== CONFIG =====================
import os, json, time, tempfile
from typing import Dict, List
from tqdm import tqdm

# ---------- KEYS ----------
OPENAI_KEY = ""
COHERE_KEY = ""

# ---------- RUNTIME ----------

# LLM_MODELS = [
#     "gpt-3.5-turbo",
#     "gpt-4o-mini",
#     "gpt-4o",
#     "gpt-4-turbo",
#     "cohere-command-r",
#     "cohere-command-r-plus"
# ]

# LLM_MODELS = [
#     "gpt-3.5-turbo",
#     "gpt-4o-mini"
# ]

DATASETS = [
    "trec-covid",
    "nfcorpus",
    "webis-touche2020",
    "dbpedia-entity",
    "scifact",
    "signal1m",
    "trec-news",
    "robust04",
    "dl19",
    "dl20"
]

LLM_MODELS = ["gpt-4o"]

# LLM_MODELS = ["gpt-4o_full", "gpt-4o-mini_full"]
### if we want to use _full uncommnet 231 otherwise comment it)
# LLM_MODELS = ["gpt-4o-mini_full"]

MODE = 2

TEST_SINGLE_QUERY = False
PRINT_SINGLE_NDCG = False
SAVE_ROOT = "logs_llm_rerank"
os.makedirs(SAVE_ROOT, exist_ok=True)

# ---------- IMPORTS ----------
try:
    from pyserini.search.lucene import LuceneSearcher
except Exception:
    from pyserini.search import LuceneSearcher
try:
    from pyserini.search import get_topics, get_qrels
except Exception:
    from pyserini.search._base import get_topics, get_qrels

BEIR_ALIAS = {
    "trec-covid": ("beir-v1.0.0-trec-covid.flat", "beir-v1.0.0-trec-covid-test"),
    "nfcorpus": ("beir-v1.0.0-nfcorpus.flat", "beir-v1.0.0-nfcorpus-test"),
    "webis-touche2020": ("beir-v1.0.0-webis-touche2020.flat", "beir-v1.0.0-webis-touche2020-test"),
    "dbpedia-entity": ("beir-v1.0.0-dbpedia-entity.flat", "beir-v1.0.0-dbpedia-entity-test"),
    "scifact": ("beir-v1.0.0-scifact.flat", "beir-v1.0.0-scifact-test"),
    "signal1m": ("beir-v1.0.0-signal1m.flat", "beir-v1.0.0-signal1m-test"),
    "trec-news": ("beir-v1.0.0-trec-news.flat", "beir-v1.0.0-trec-news-test"),
    "robust04": ("beir-v1.0.0-robust04.flat", "beir-v1.0.0-robust04-test"),
}
MSMARCO_ALIAS = {
    "dl19": ("msmarco-v1-passage", "dl19-passage"),
    "dl20": ("msmarco-v1-passage", "dl20-passage"),
}

from rank_gpt import sliding_windows
from trec_eval import EvalFunction
from retrieve_bm25 import retrieve_topk

# ==================================================

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def _openai_chat(model: str, messages: List[Dict]) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_KEY)
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
    return resp.choices[0].message.content

def _cohere_rerank(query: str, passages: List[str], model: str) -> List[int]:
    import cohere
    co = cohere.Client(COHERE_KEY)
    out = co.rerank(model=model, query=query, documents=passages, top_n=len(passages))
    return [r.index for r in out.results]

def _iter_topics_in_order(topics_dict: Dict) -> List[str]:
    qids = list(topics_dict.keys())
    try:
        qids = sorted(qids, key=lambda x: int(x))
    except Exception:
        qids = sorted(map(str, qids))
    return list(map(str, qids))

def _save_log(dataset: str, qid: str, query: str, input_docs: List[Dict], output_docs: List[Dict], model_name: str):
    out_dir = os.path.join(SAVE_ROOT, dataset, model_name)
    ensure_dir(out_dir)
    input_ids = [d.get("docid", "") for d in input_docs]
    output_ids = [d.get("docid", "") for d in output_docs]
    with open(os.path.join(out_dir, f"qid_{qid}.json"), "w") as f:
        json.dump({
            "qid": qid,
            "query": query,
            "input_docs": input_docs,
            "output_docs": output_docs,
            "input_docids": input_ids,
            "output_docids": output_ids,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

# def _load_top30_from_gpt35_logs(dataset: str, qid: str) -> List[Dict]:
#     path = os.path.join(SAVE_ROOT, dataset, "gpt-3.5-turbo", f"qid_{qid}.json")
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Missing GPT-3.5 log for qid={qid}: {path}")
#     with open(path) as f:
#         obj = json.load(f)
#     return obj["output_docs"][:30]


def _load_top30_from_gpt35_logs(dataset: str, qid: str, searcher=None) -> list:
    path = os.path.join(SAVE_ROOT, dataset, "gpt-3.5-turbo", f"qid_{qid}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing GPT-3.5 log for qid={qid}: {path}")
    with open(path) as f:
        obj = json.load(f)
    top30 = obj.get("output_docs", [])[:30]

    # ✅ Fix: recover text from 'content' or from Lucene if available
    for d in top30:
        # Use existing text if present
        text = d.get("text", "")
        # Fall back to 'content' (stored as a JSON string in the GPT-3.5 log)
        if not text and "content" in d:
            text = d["content"]
        # Fall back to Lucene index if still empty
        if not text and searcher is not None:
            doc = searcher.doc(d["docid"])
            if doc is not None:
                text = doc.raw() or doc.stored_fields().get("contents", "")
        d["text"] = text

    return top30


def _evaluate_and_print(dataset_topic_name, run_items, label, summary_lines=None):
    tmp = tempfile.NamedTemporaryFile(delete=False).name
    EvalFunction.write_file(run_items, tmp)
    ndcg = EvalFunction.main(dataset_topic_name, tmp)
    ndcg10 = None
    if isinstance(ndcg, dict):
        ndcg10 = ndcg.get("NDCG@10", None)
        if ndcg10 is not None:
            print(f"[{label}] Mean nDCG@10 = {ndcg10:.4f}")
        else:
            print(f"[{label}] Evaluation metrics: {ndcg}")
    else:
        ndcg10 = ndcg
        print(f"[{label}] Mean nDCG@10 = {ndcg:.4f}")
    if summary_lines is not None and ndcg10 is not None:
        summary_lines.append(f"{label}: nDCG@10 = {ndcg10:.4f}")
    return ndcg10

def run_llm_rerank(dataset: str, model_name: str, mode: int = 1, summary_lines=None):
    if dataset in MSMARCO_ALIAS:
        index_name, topics_name = MSMARCO_ALIAS[dataset]
    elif dataset in BEIR_ALIAS:
        index_name, topics_name = BEIR_ALIAS[dataset]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"\n[INFO] === {dataset} | index={index_name} | topics={topics_name} | model={model_name} | mode={mode} ===")

    bm25_run, qrels, topics, searcher = retrieve_topk(dataset, k=100)

    if isinstance(topics, list):
        tmp = {}
        for i, t in enumerate(topics):
            if isinstance(t, dict):
                qid = str(t.get("id", i))
                tmp[qid] = t.get("title") or t.get("text") or t.get("query") or str(t)
            else:
                tmp[str(i)] = str(t)
        topics = tmp
    elif isinstance(topics, dict):
        topics = {str(k): v for k, v in topics.items()}

    ordered_qids = _iter_topics_in_order(topics)

    normalized_bm25 = []
    if isinstance(bm25_run, dict):
        for qid, hits in bm25_run.items():
            normalized_hits = []
            for h in hits:
                if isinstance(h, tuple):
                    docid, score = h[0], h[1]
                elif isinstance(h, dict):
                    docid, score = h.get("docid", ""), h.get("score", 0.0)
                else:
                    continue
                normalized_hits.append({"qid": str(qid), "docid": str(docid), "score": float(score)})
            normalized_bm25.append({"query": str(topics.get(str(qid), "")), "hits": normalized_hits})
    elif isinstance(bm25_run, list):
        for item in bm25_run:
            if "hits" in item:
                normalized_bm25.append(item)

    _evaluate_and_print(topics_name, normalized_bm25, "BM25", summary_lines)

    first_qid = ordered_qids[0] if ordered_qids else None
    results = []
    iterator = tqdm(ordered_qids, desc=f"{dataset}-{model_name}", ncols=100)

    for qid in iterator:
        if TEST_SINGLE_QUERY and qid != first_qid:
            continue
        qtext = topics.get(qid)
        if qtext is None:
            continue

        candidates = []

        # ---- Candidate selection ----
        if mode == 1:
            for r in normalized_bm25:
                hits = r.get("hits", [])
                for h in hits:
                    if str(h.get("qid")) == str(qid):
                        doc = searcher.doc(h["docid"])
                        if doc is None: continue
                        candidates.append({"qid": qid, "docid": h["docid"], "score": h["score"], "text": doc.raw()})

        elif mode == 2:
            # Cohere models always use BM25 docs directly
            if "cohere" in model_name or "command" in model_name:
                for r in normalized_bm25:
                    hits = r.get("hits", [])
                    for h in hits:
                        if str(h.get("qid")) == str(qid):
                            doc = searcher.doc(h["docid"])
                            if doc is None: continue
                            candidates.append({"qid": qid, "docid": h["docid"], "score": h["score"], "text": doc.raw()})
            else:
                try:
#                     top30 = _load_top30_from_gpt35_logs(dataset, qid)
                    top30 = _load_top30_from_gpt35_logs(dataset, qid, searcher=searcher)
                    candidates = [{"qid": qid, "docid": d["docid"], "score": d.get("score", 0.0), "text": d.get("text", "")} for d in top30]
                except FileNotFoundError:
                    if model_name == "gpt-3.5-turbo":
                        print(f"[INFO] GPT-3.5 logs not found; running fresh rerank for {qid}")
                        for r in normalized_bm25:
                            hits = r.get("hits", [])
                            for h in hits:
                                if str(h.get("qid")) == str(qid):
                                    doc = searcher.doc(h["docid"])
                                    if doc is None: continue
                                    candidates.append({"qid": qid, "docid": h["docid"], "score": h["score"], "text": doc.raw()})
                    else:
                        print(f"[WARN] GPT-3.5 logs missing for qid={qid}. Skipping.")
                        continue
        else:
            raise ValueError("mode must be 1 or 2")

        if not candidates:
            continue

        # ---- Reranking ----
        if model_name.startswith("gpt-"):
            rankgpt_hits = [{"qid": str(c["qid"]), "docid": str(c["docid"]), "content": str(c["text"]), "score": float(c["score"])} for c in candidates]
            if model_name.strip().lower() == "gpt-4":
                for c in candidates:
                    if len(c.get("text", "")) > 1000:
                        c["text"] = c["text"][:1000] + " ..."
                        
#             api_model_name = model_name.replace("_full", "")
#             model_name=api_model_name
            
            new_item = sliding_windows(item={"query": qtext, "hits": rankgpt_hits}, rank_start=0, rank_end=len(rankgpt_hits), window_size=20, step=10, model_name=model_name, api_key=OPENAI_KEY)
            
            reranked_hits = new_item.get("hits", [])
            
        elif "cohere" in model_name or "command" in model_name:
            passages = [c["text"] for c in candidates]
            order = _cohere_rerank(qtext, passages, model=model_name)
            reranked_hits = [candidates[i] for i in order]
        else:
            raise ValueError(f"Unsupported LLM model: {model_name}")

        _save_log(dataset, str(qid), qtext, candidates, reranked_hits, model_name)
        results.append({"query": qtext, "hits": [{"qid": str(qid), "docid": h["docid"], "score": float(h["score"])} for h in reranked_hits]})

    if not TEST_SINGLE_QUERY:
        _evaluate_and_print(topics_name, results, f"{model_name}", summary_lines)

if __name__ == "__main__":
    summary_file = os.path.join(SAVE_ROOT, "summary_ndcg.txt")
    summary_lines = []
    for dset in DATASETS:
        for model in LLM_MODELS:
            run_llm_rerank(dset, model, mode=MODE, summary_lines=summary_lines)
    with open(summary_file, "w") as f:
        f.write("\n".join(summary_lines))
    print("\n[SUMMARY] nDCG@10 results written to", summary_file)

