OPENAI_API_KEY = ""


# ============================================================
# run_trend_by_source_SDCG.py
# Compare candidate sources: BM25 vs RM3 vs SPLADE
# Supports: BEIR datasets + DL19/DL20 (MS MARCO passage)
# Candidate sizes (SIZES)
# Caches candidates + rerank outputs so reruns skip expensive work
# Computes:
#  - nDCG@10 with constant IDCG using max-grade per dataset, gain=2^rel-1
#  - RBO between initial rankings of different sources
#  - significance tests (paired t-test, Wilcoxon, sign test) between sources
# ============================================================


import os
import json
import time
import socket
import math
import hashlib
from typing import Dict, List, Tuple, Any, Optional
from itertools import combinations

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from pyserini.search import get_topics, get_qrels
from pyserini.search.lucene import LuceneSearcher, LuceneImpactSearcher
from pyserini.encode import SpladeQueryEncoder

# ------------------------------
# Your modules (unchanged)
# ------------------------------
from config import (
    MONOBERT_CKPT,
    MONOT5_BASE_CKPT,
    MONOT5_3B_CKPT,
    MMARCO_CE_CKPT,
)

from rerank_supervised import MonoBERTReRanker, MonoT5ReRanker, rerank_with_progress as rerank_supervised
from rerank_unsupervised import CrossEncoderReRanker, rerank_with_progress as rerank_unsup
from rank_gpt import sliding_windows


# ============================================================
# CONFIG
# ============================================================

# Example set; add more freely (BEIR names + "dl19" + "dl20")
# DATASETS_ALL = [
#     "trec-covid",
#     "nfcorpus",
#     "webis-touche2020",
#     "dbpedia-entity",
#     "scifact",
#     "signal1m",
#     "trec-news",
#     "robust04",
#     "dl19",
#     "dl20",
# ]

# METHODS_ALL = [
#     "MonoBERT",
#     "MonoT5_220M",
#     "GPT35",
#     "GPT4oMini",
# ]

INPUT_SOURCES_ALL = ["BM25", "RM3", "SPLADE"]

# ---- choose here ----
# DATASETS = ["dl19"]
# SIZES = [20, 100]
# INPUT_SOURCES = ["BM25", "RM3", "SPLADE"]
# METHODS = ["MonoBERT"]


DATASETS = [
#     "dl19",
#     "trec-covid",
#     "webis-touche2020",
    "trec-news",
]
SIZES = [20, 100]
INPUT_SOURCES = ["BM25", "RM3", "SPLADE"]
METHODS = [
    "MonoBERT",
    "MonoT5_220M",
#     "MonoT5_3B",
    "GPT35",
    "GPT4oMini",
]
# ---------------------

# ---------- OpenAI key ----------
# Env var overrides only if set; otherwise falls back to hardcoded.
OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY", "").strip() or OPENAI_API_KEY.strip())

# RM3 parameters (common defaults)
RM3_FB_TERMS = 10
RM3_FB_DOCS = 10
RM3_ORIG_Q_WEIGHT = 0.5

# SPLADE query encoder name
# ⚠️ must be a valid HuggingFace model repo id (public) for your environment
SPLADE_QUERY_ENCODER_NAME = "naver/splade-cocondenser-ensembledistil"

# Evaluation config for new nDCG
TOP_K_EVAL = 10  # nDCG@10
GAIN_POWER = True  # gain = 2^rel - 1 (trec_eval style)
RBO_P = 0.9       # RBO persistence parameter
SIGNIFICANCE_ALPHA = 0.05

# Caching / rerun control
FORCE_RERUN = os.environ.get("FORCE_RERUN", "0").strip() == "1"


# ============================================================
# OUTPUT ROOTS (separate from by_size) + per-run subfolders
# ============================================================

RESULTS_ROOT = "./results_by_source"
LOGS_ROOT    = "./logs_llm_rerank_sources"
PLOTS_ROOT   = "./plots_by_source"  # legacy; per-run plots go under RESULTS_ROOT

os.makedirs(RESULTS_ROOT, exist_ok=True)
os.makedirs(LOGS_ROOT, exist_ok=True)
os.makedirs(PLOTS_ROOT, exist_ok=True)

# Run tag to avoid overwriting when you run multiple terminals
def _make_run_tag() -> str:
    env_tag = os.environ.get("RUN_TAG", "").strip()
    if env_tag:
        return env_tag
    ts = time.strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    host = socket.gethostname().split(".")[0]
    return f"{ts}_{host}_pid{pid}"

RUN_TAG = _make_run_tag()
RUN_ROOT = os.path.join(RESULTS_ROOT, f"run_{RUN_TAG}")
RUN_PLOTS_ROOT = os.path.join(RUN_ROOT, "plots")
RUN_LOG_TXT = os.path.join(RUN_ROOT, "run_log.txt")
RUN_RESULTS_CSV = os.path.join(RUN_ROOT, "results_long.csv")
RUN_RESULTS_WIDE_CSV = os.path.join(RUN_ROOT, "results_wide.csv")
RUN_MAX_GRADE_JSON = os.path.join(RUN_ROOT, "max_grade_per_dataset.json")
RUN_SIGNIF_DIR = os.path.join(RUN_ROOT, "significance")
RUN_RBO_DIR = os.path.join(RUN_ROOT, "rbo_between_sources")

for p in [RUN_ROOT, RUN_PLOTS_ROOT, RUN_SIGNIF_DIR, RUN_RBO_DIR]:
    os.makedirs(p, exist_ok=True)

# Deterministic cache root (shared across runs)
CACHE_ROOT = os.path.join(LOGS_ROOT, "cache")
os.makedirs(CACHE_ROOT, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def log_line(fp, msg: str):
    fp.write(msg.rstrip() + "\n")
    fp.flush()

def atomic_write_text(path: str, text: str):
    ensure_dir(os.path.dirname(path))
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)

def atomic_write_json(path: str, obj: Any, indent: int = 2):
    ensure_dir(os.path.dirname(path))
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)

def stable_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def is_msmarco_dl(dataset: str) -> bool:
    return dataset.lower() in ["dl19", "dl20"]

def topics_name_for(dataset: str) -> str:
    d = dataset.lower()
    if d == "dl19":
        return "dl19-passage"
    if d == "dl20":
        return "dl20-passage"
    return f"beir-v1.0.0-{dataset}-test"

def bm25_index_candidates(dataset: str) -> List[str]:
    if is_msmarco_dl(dataset):
        return ["msmarco-v1-passage"]
    return [f"beir-v1.0.0-{dataset}.flat"]

def splade_index_candidates(dataset: str) -> List[str]:
    if is_msmarco_dl(dataset):
        return [
            "msmarco-v1-passage.splade-pp-ed",
            "msmarco-v1-passage.splade-pp",
            "msmarco-v1-passage.splade",
        ]
    base = f"beir-v1.0.0-{dataset}"
    return [
        f"{base}.splade-pp-ed",
        f"{base}.splade-pp",
        f"{base}.splade",
    ]

def open_prebuilt_lucene_index(index_names: List[str]) -> Tuple[LuceneSearcher, str]:
    last_err = None
    for name in index_names:
        try:
            return LuceneSearcher.from_prebuilt_index(name), name
        except Exception as e:
            last_err = e
    raise RuntimeError(
        "Could not open any Lucene prebuilt index from candidates:\n"
        + "\n".join([f"  - {n}" for n in index_names])
        + f"\nLast error: {repr(last_err)}"
    )

def open_prebuilt_impact_index(index_names: List[str], query_encoder) -> Tuple[LuceneImpactSearcher, str]:
    last_err = None
    for name in index_names:
        try:
            return LuceneImpactSearcher.from_prebuilt_index(name, query_encoder), name
        except Exception as e:
            last_err = e
    raise RuntimeError(
        "Could not open any Impact (SPLADE) prebuilt index from candidates:\n"
        + "\n".join([f"  - {n}" for n in index_names])
        + f"\nLast error: {repr(last_err)}"
    )

def get_doc_text(searcher, docid: str) -> str:
    d = searcher.doc(docid)
    if d is None:
        return ""
    raw = d.raw()
    if raw is None:
        raw = ""
    try:
        obj = json.loads(raw)
        title = obj.get("title", "")
        text  = obj.get("text", "") or obj.get("contents", "")
        if title and text:
            return f"{title}. {text}"
        return title or text or raw
    except Exception:
        return raw

def normalize_topics_to_queries(topics: Any) -> Dict[str, str]:
    queries: Dict[str, str] = {}
    if isinstance(topics, dict):
        for qid, q in topics.items():
            if isinstance(q, dict):
                qtext = q.get("title") or q.get("text") or ""
                if not qtext and len(q) > 0:
                    qtext = str(list(q.values())[0])
            else:
                qtext = str(q)
            queries[str(qid)] = str(qtext)
            try:
                queries[str(int(qid))] = str(qtext)
            except Exception:
                pass
    else:
        for i, q in enumerate(topics):
            queries[str(i)] = str(q)
    return queries

def normalize_qrels(qrels: Dict[Any, Dict[Any, Any]], dataset: str) -> Dict[str, Dict[str, int]]:
    """Normalize qrels into {qid_str: {docid_str: int_rel}}, adding safe aliases.

    This fixes common mismatches, especially for MS MARCO passage (dl19/dl20),
    where docids may appear with/without the 'msmarco_passage_' prefix depending
    on the qrels/index variant.
    """
    out: Dict[str, Dict[str, int]] = {}
    msm = is_msmarco_dl(dataset)
    prefix = "msmarco_passage_"

    for qid, docrels in (qrels or {}).items():
        qid_str = str(qid)
        out.setdefault(qid_str, {})
        for docid, rel in (docrels or {}).items():
            try:
                r = int(rel)
            except Exception:
                try:
                    r = int(float(rel))
                except Exception:
                    r = 0

            did = str(docid)
            out[qid_str][did] = r

            # Add MS MARCO passage aliases
            if msm:
                if did.startswith(prefix):
                    out[qid_str].setdefault(did[len(prefix):], r)
                elif did.isdigit():
                    out[qid_str].setdefault(prefix + did, r)

        # Add int-string alias for qids when possible
        try:
            out.setdefault(str(int(qid_str)), out[qid_str])
        except Exception:
            pass

    return out

# ============================================================
# New nDCG: constant ideal DCG at depth TOP_K_EVAL
# Gain = 2^rel - 1 if GAIN_POWER else rel
# IDCG_const = sum_{i=1..TOP_K_EVAL} gain(max_grade) / log2(i+1)
# ============================================================

def max_grade_from_qrels(qrels: Dict[str, Dict[str, int]]) -> int:
    mx = 0
    for _, docrels in qrels.items():
        for _, r in docrels.items():
            try:
                rr = int(r)
            except Exception:
                rr = 0
            if rr > mx:
                mx = rr
    return int(mx)

def gain(rel: int) -> float:
    if GAIN_POWER:
        return float((2 ** int(rel)) - 1)
    return float(rel)

def idcg_constant(max_grade: int, k: int = TOP_K_EVAL) -> float:
    if k <= 0:
        return 0.0
    g = gain(int(max_grade))
    s = 0.0
    for i in range(k):
        s += g / math.log2(i + 2)
    return float(s)

def get_ranked_docids_from_run(run_for_qid: Dict[str, float], k: int) -> List[str]:
    # sort by score descending; tie-break by docid for stability
    items = sorted(run_for_qid.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
    return [str(docid) for docid, _ in items[:k]]

def dcg_from_ranked_docids(ranked_docids: List[str], qrels_for_qid: Dict[str, int], k: int = TOP_K_EVAL) -> float:
    s = 0.0
    for i, docid in enumerate(ranked_docids[:k]):
        rel = int(qrels_for_qid.get(str(docid), 0))
        s += gain(rel) / math.log2(i + 2)
    return float(s)

def per_query_ndcg10_constant(run: Dict[str, Dict[str, float]], qrels: Dict[str, Dict[str, int]],
                             idcg_const: float, k_eval: int = TOP_K_EVAL) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for qid, run_for_qid in run.items():
        qid_str = str(qid)
        qrels_for_qid = qrels.get(qid_str, {})
        ranked = get_ranked_docids_from_run(run_for_qid, k_eval)
        dcg = dcg_from_ranked_docids(ranked, qrels_for_qid, k_eval)
        out[qid_str] = 0.0 if idcg_const <= 0 else float(dcg / idcg_const)
    return out

def mean_of_dict(d: Dict[str, float]) -> float:
    if not d:
        return 0.0
    return float(sum(d.values()) / len(d))

def run_from_docids(docids_by_qid: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    # make descending artificial scores so ranking is preserved
    run = {}
    for qid, docids in docids_by_qid.items():
        n = len(docids)
        run[qid] = {str(d): float(n - i) for i, d in enumerate(docids)}
    return run

def run_from_candidates(candidates: Dict[str, List[Tuple[str, float]]], k: int) -> Dict[str, Dict[str, float]]:
    run = {}
    for qid, items in candidates.items():
        run[str(qid)] = {str(docid): float(score) for (docid, score) in items[:k]}
    return run


# ============================================================
# RBO between two ranked lists
# ============================================================

def rbo_score(list1: List[str], list2: List[str], p: float = RBO_P, k: Optional[int] = None) -> float:
    """
    Finite-depth RBO (extrapolated) as in Webber et al.
    If k is None, uses min(len(list1), len(list2)).
    """
    if not list1 and not list2:
        return 1.0
    if not list1 or not list2:
        return 0.0

    if k is None:
        k = min(len(list1), len(list2))
    k = int(min(k, len(list1), len(list2)))
    if k <= 0:
        return 0.0

    s1, s2 = set(), set()
    overlap = 0
    summation = 0.0

    for d in range(1, k + 1):
        s1.add(list1[d - 1])
        s2.add(list2[d - 1])
        overlap = len(s1.intersection(s2))
        summation += overlap / d * (p ** (d - 1))

    rbo = (1 - p) * summation
    return float(rbo)


# ============================================================
# Significance tests between paired per-query metric lists
# ============================================================

def paired_tests(a: List[float], b: List[float]) -> Tuple[float, float, float, int, int, int, int]:
    """
    Returns: p_ttest, p_wilcoxon, p_signtest, n_effective, pos, neg, ties
    """
    assert len(a) == len(b)
    diffs = [ai - bi for ai, bi in zip(a, b)]
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    ties = sum(1 for d in diffs if d == 0)
    n_eff = pos + neg

    p_ttest = float("nan")
    p_wilcoxon = float("nan")
    p_signtest = float("nan")

    try:
        from scipy.stats import ttest_rel, wilcoxon, binomtest
        if n_eff >= 2:
            p_ttest = float(ttest_rel(a, b).pvalue)
        if n_eff >= 1:
            try:
                p_wilcoxon = float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
            except Exception:
                p_wilcoxon = float("nan")
        if n_eff >= 1:
            p_signtest = float(binomtest(k=pos, n=n_eff, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        # If scipy missing, leave NaNs; still return counts
        pass

    return p_ttest, p_wilcoxon, p_signtest, n_eff, pos, neg, ties


# ============================================================
# Cache paths
# ============================================================

def cache_path_candidates(dataset: str, source: str, k: int) -> str:
    return os.path.join(CACHE_ROOT, f"k{k}", dataset, source, "candidates.json")

def cache_path_candidates_meta(dataset: str, source: str, k: int) -> str:
    return os.path.join(CACHE_ROOT, f"k{k}", dataset, source, "meta.json")

def cache_path_run_docids(dataset: str, source: str, k: int, method: str) -> str:
    return os.path.join(CACHE_ROOT, f"k{k}", dataset, source, method, "output_docids.json")

def cache_path_perquery_jsonl(dataset: str, source: str, k: int, method: str) -> str:
    return os.path.join(CACHE_ROOT, f"k{k}", dataset, source, method, "per_query.jsonl")

def cache_path_perquery_ndcg_csv(dataset: str, source: str, k: int, method: str) -> str:
    return os.path.join(CACHE_ROOT, f"k{k}", dataset, source, method, "per_query_ndcg.csv")


# ============================================================
# Candidate retrieval by source (with caching)
# ============================================================

def retrieve_candidates(dataset: str, source: str, k: int):
    """
    Returns:
      candidates: {qid -> [(docid, score), ... up to k]}
      qrels
      topics
      searcher
      topics_name_used
      index_name_used
    """
    source = source.upper()
    cpath = cache_path_candidates(dataset, source, k)
    mpath = cache_path_candidates_meta(dataset, source, k)

    if (not FORCE_RERUN) and os.path.exists(cpath) and os.path.exists(mpath):
        with open(cpath, "r") as f:
            candidates = json.load(f)
        with open(mpath, "r") as f:
            meta = json.load(f)
        topics_name_used = meta["topics_name"]
        index_name_used = meta["index_name"]
        if meta.get("source", "").upper() != source:
            raise RuntimeError("Cache source mismatch.")
        topics = get_topics(topics_name_used)
        qrels = normalize_qrels(get_qrels(topics_name_used), dataset)
        searcher = None

        if source in ["BM25", "RM3"]:
            searcher, _ = open_prebuilt_lucene_index([index_name_used])
            if source == "RM3":
                searcher.set_rm3(RM3_FB_TERMS, RM3_FB_DOCS, RM3_ORIG_Q_WEIGHT)
        else:
            query_encoder = SpladeQueryEncoder(SPLADE_QUERY_ENCODER_NAME)
            searcher, _ = open_prebuilt_impact_index([index_name_used], query_encoder)

        return candidates, qrels, topics, searcher, topics_name_used, index_name_used

    # ---- compute fresh ----
    if source in ["BM25", "RM3"]:
        searcher, index_name_used = open_prebuilt_lucene_index(bm25_index_candidates(dataset))

        if source == "RM3":
            searcher.set_rm3(RM3_FB_TERMS, RM3_FB_DOCS, RM3_ORIG_Q_WEIGHT)

        topics_name_used = topics_name_for(dataset)
        topics = get_topics(topics_name_used)
        qrels = normalize_qrels(get_qrels(topics_name_used), dataset)

    elif source == "SPLADE":
        query_encoder = SpladeQueryEncoder(SPLADE_QUERY_ENCODER_NAME)
        searcher, index_name_used = open_prebuilt_impact_index(splade_index_candidates(dataset), query_encoder)

        topics_name_used = topics_name_for(dataset)
        topics = get_topics(topics_name_used)
        qrels = normalize_qrels(get_qrels(topics_name_used), dataset)

    else:
        raise ValueError(f"Unknown input source: {source}")

    queries = normalize_topics_to_queries(topics)

    # restrict to qids that appear in qrels
    qrels_qids = set(str(q) for q in qrels.keys())
    for q in list(qrels_qids):
        try:
            qrels_qids.add(str(int(q)))
        except Exception:
            pass
    filtered_qids = [qid for qid in queries.keys() if str(qid) in qrels_qids]
    if len(filtered_qids) == 0:
        filtered_qids = list(queries.keys())

    try:
        qids = sorted(set(filtered_qids), key=lambda x: int(x) if str(x).isdigit() else str(x))
    except Exception:
        qids = sorted(set(filtered_qids), key=str)

    candidates: Dict[str, List[Tuple[str, float]]] = {}
    it = tqdm(qids, desc=f"[RETRIEVE] {dataset} | {source} | top-{k}", ncols=110)

    for qid in it:
        query = queries[qid]
        hits = searcher.search(query, k)
        candidates[str(qid)] = [(str(h.docid), float(h.score)) for h in hits]

    atomic_write_json(cpath, candidates)
    atomic_write_json(mpath, {
        "dataset": dataset,
        "source": source,
        "k": k,
        "topics_name": topics_name_used,
        "index_name": index_name_used,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    return candidates, qrels, topics, searcher, topics_name_used, index_name_used


# ============================================================
# Rerankers (produce output docids list per qid)
# ============================================================

def rerank_supervised_docids(method: str, candidates, queries, doc_fetcher, batch_size=8) -> Dict[str, List[str]]:
    if method == "MonoBERT":
        scorer = MonoBERTReRanker(MONOBERT_CKPT, max_len=512)
        rr = rerank_supervised("monoBERT", scorer, candidates, queries, doc_fetcher,
                               batch_size=batch_size, colour="blue")
    elif method == "MonoT5_220M":
        scorer = MonoT5ReRanker(MONOT5_BASE_CKPT)
        rr = rerank_supervised("monoT5-base", scorer, candidates, queries, doc_fetcher,
                               batch_size=batch_size, colour="blue")
    elif method == "MonoT5_3B":
        scorer = MonoT5ReRanker(MONOT5_3B_CKPT)
        rr = rerank_supervised("monoT5-3B", scorer, candidates, queries, doc_fetcher,
                               batch_size=max(1, batch_size // 2), colour="blue")
    else:
        raise ValueError(f"Unknown supervised method: {method}")

    out: Dict[str, List[str]] = {}
    for qid, pairs in rr.items():
        out[str(qid)] = [str(docid) for docid, _ in pairs]
    return out

def rerank_mmarco_ce_docids(candidates, queries, doc_fetcher, batch_size=16) -> Dict[str, List[str]]:
    scorer = CrossEncoderReRanker(MMARCO_CE_CKPT)
    rr = rerank_unsup("mMARCO-CE", scorer, candidates, queries, doc_fetcher,
                      batch_size=batch_size, colour="yellow")
    out: Dict[str, List[str]] = {}
    for qid, pairs in rr.items():
        out[str(qid)] = [str(docid) for docid, _ in pairs]
    return out

def rerank_rankgpt_llm_docids(dataset: str, candidates, topics, searcher, k: int,
                             model_name: str, openai_key: str, source: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}

    def safe_get_items(qid_str: str):
        return candidates.get(qid_str, [])

    def qtext_of(qid_str: str):
        q = ""
        if isinstance(topics, dict):
            if qid_str in topics:
                q = topics[qid_str]
            else:
                try:
                    q = topics[int(qid_str)]
                except Exception:
                    q = ""
        if isinstance(q, dict):
            return q.get("title") or q.get("text") or str(list(q.values())[0])
        return str(q)

    qids = sorted(candidates.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    iterator = tqdm(qids, desc=f"{dataset}-{source}-{model_name}-k{k}", ncols=110)

    for qid in iterator:
        qid_str = str(qid)
        query = qtext_of(qid_str)

        items = safe_get_items(qid_str)[:k]
        if not items:
            continue

        hits = []
        for docid, score in items:
            docid = str(docid)
            text = get_doc_text(searcher, docid)
            if not isinstance(text, str):
                text = ""
            text = text.strip()
            if not text:
                text = f"(no content for docid {docid})"
            hits.append({
                "qid": qid_str,
                "docid": docid,
                "score": float(score),
                "content": text,
            })

        new_item = sliding_windows(
            item={"query": query, "hits": hits},
            rank_start=0,
            rank_end=len(hits),
            window_size=min(20, len(hits)),
            step=10,
            model_name=model_name,
            api_key=openai_key
        )

        reranked_hits = new_item.get("hits", [])
        out_docids = [str(h["docid"]) for h in reranked_hits]
        out[qid_str] = out_docids

    return out


# ============================================================
# Cache-aware runner for docids + per-query logs
# ============================================================

def load_docids_if_cached(dataset: str, source: str, k: int, method: str) -> Optional[Dict[str, List[str]]]:
    p = cache_path_run_docids(dataset, source, k, method)
    if (not FORCE_RERUN) and os.path.exists(p):
        with open(p, "r") as f:
            obj = json.load(f)
        return {str(qid): [str(d) for d in lst] for qid, lst in obj.items()}
    return None

def save_docids_cache(dataset: str, source: str, k: int, method: str, docids_by_qid: Dict[str, List[str]]):
    p = cache_path_run_docids(dataset, source, k, method)
    atomic_write_json(p, docids_by_qid)

def save_per_query_logs(dataset: str, source: str, k: int, method: str,
                        qid_to_query: Dict[str, str],
                        input_docids_by_qid: Dict[str, List[str]],
                        output_docids_by_qid: Dict[str, List[str]],
                        per_query_ndcg: Dict[str, float]):
    jsonl_path = cache_path_perquery_jsonl(dataset, source, k, method)
    csv_path = cache_path_perquery_ndcg_csv(dataset, source, k, method)

    rows = []
    lines = []
    for qid in sorted(per_query_ndcg.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        rec = {
            "dataset": dataset,
            "source": source,
            "k": k,
            "method": method,
            "qid": str(qid),
            "query": qid_to_query.get(str(qid), ""),
            "ndcg@10": float(per_query_ndcg[qid]),
            "input_docids": input_docids_by_qid.get(str(qid), []),
            "output_docids": output_docids_by_qid.get(str(qid), []),
        }
        lines.append(json.dumps(rec))
        rows.append({
            "dataset": dataset,
            "source": source,
            "k": k,
            "method": method,
            "qid": str(qid),
            "ndcg@10": float(per_query_ndcg[qid]),
        })

    # write cleanly (overwrite ok; deterministic)
    atomic_write_text(jsonl_path, "\n".join(lines) + ("\n" if lines else ""))
    pd.DataFrame(rows).to_csv(csv_path, index=False)

def get_or_compute_method_docids(dataset: str, source: str, k: int, method: str,
                                 candidates: Dict[str, List[Tuple[str, float]]],
                                 topics: Any,
                                 searcher,
                                 doc_fetcher,
                                 queries_map: Dict[str, str]) -> Dict[str, List[str]]:
    cached = load_docids_if_cached(dataset, source, k, method)
    if cached is not None:
        return cached

    # compute fresh
    if method == "RETRIEVER":
        out = {qid: [str(docid) for docid, _ in items[:k]] for qid, items in candidates.items()}
    elif method in ["MonoBERT", "MonoT5_220M", "MonoT5_3B"]:
        out = rerank_supervised_docids(method, candidates, queries_map, doc_fetcher, batch_size=8 if method != "MonoT5_3B" else 4)
    elif method == "mMARCO_CE":
        out = rerank_mmarco_ce_docids(candidates, queries_map, doc_fetcher, batch_size=16)
    elif method == "GPT35":
        out = rerank_rankgpt_llm_docids(dataset, candidates, topics, searcher, k, "gpt-3.5-turbo", OPENAI_API_KEY, source)
    elif method == "GPT4oMini":
        out = rerank_rankgpt_llm_docids(dataset, candidates, topics, searcher, k, "gpt-4o-mini", OPENAI_API_KEY, source)
    else:
        raise ValueError(f"Unknown method: {method}")

    save_docids_cache(dataset, source, k, method, out)
    return out


# ============================================================
# RBO + Significance computation and saving
# ============================================================

def compute_and_save_rbo(dataset: str, k: int, retriever_docids_by_source: Dict[str, Dict[str, List[str]]]):
    sources = sorted(retriever_docids_by_source.keys())
    rows = []
    for a, b in combinations(sources, 2):
        qa = retriever_docids_by_source[a]
        qb = retriever_docids_by_source[b]
        common_qids = sorted(set(qa.keys()).intersection(qb.keys()), key=lambda x: int(x) if str(x).isdigit() else str(x))
        for qid in common_qids:
            la = qa[qid][:k]
            lb = qb[qid][:k]
            rows.append({
                "dataset": dataset,
                "k": k,
                "source_A": a,
                "source_B": b,
                "qid": str(qid),
                "rbo": rbo_score(la, lb, p=RBO_P, k=min(len(la), len(lb))),
            })
    df = pd.DataFrame(rows)
    out_csv = os.path.join(RUN_RBO_DIR, f"rbo_{dataset}_k{k}.csv")
    df.to_csv(out_csv, index=False)

    summary_rows = []
    if not df.empty:
        for (a, b), g in df.groupby(["source_A", "source_B"]):
            summary_rows.append({
                "dataset": dataset,
                "k": k,
                "source_A": a,
                "source_B": b,
                "mean_rbo": float(g["rbo"].mean()),
                "n": int(len(g)),
            })
    df_sum = pd.DataFrame(summary_rows)
    out_sum_csv = os.path.join(RUN_RBO_DIR, f"rbo_summary_{dataset}_k{k}.csv")
    df_sum.to_csv(out_sum_csv, index=False)

    return out_csv, out_sum_csv

def compute_and_save_significance(dataset: str, k: int, method: str,
                                  per_query_ndcg_by_source: Dict[str, Dict[str, float]]):
    sources = sorted(per_query_ndcg_by_source.keys())
    rows = []
    for a, b in combinations(sources, 2):
        da = per_query_ndcg_by_source[a]
        db = per_query_ndcg_by_source[b]
        common_qids = sorted(set(da.keys()).intersection(db.keys()), key=lambda x: int(x) if str(x).isdigit() else str(x))
        a_vals = [float(da[q]) for q in common_qids]
        b_vals = [float(db[q]) for q in common_qids]

        p_t, p_w, p_s, n_eff, pos, neg, ties = paired_tests(a_vals, b_vals)
        mean_a = float(sum(a_vals) / len(a_vals)) if a_vals else 0.0
        mean_b = float(sum(b_vals) / len(b_vals)) if b_vals else 0.0

        rows.append({
            "dataset": dataset,
            "k": k,
            "method": method,
            "source_A": a,
            "source_B": b,
            "mean_A": mean_a,
            "mean_B": mean_b,
            "p_ttest": p_t,
            "p_wilcoxon": p_w,
            "p_signtest": p_s,
            "ttest_sig": "SIGNIFICANT" if (not math.isnan(p_t) and p_t < SIGNIFICANCE_ALPHA) else "NOT SIGNIFICANT",
            "wilcoxon_sig": "SIGNIFICANT" if (not math.isnan(p_w) and p_w < SIGNIFICANCE_ALPHA) else "NOT SIGNIFICANT",
            "signtest_sig": "SIGNIFICANT" if (not math.isnan(p_s) and p_s < SIGNIFICANCE_ALPHA) else "NOT SIGNIFICANT",
            "n_effective": n_eff,
            "pos": pos,
            "neg": neg,
            "ties": ties,
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RUN_SIGNIF_DIR, f"significance_{dataset}_k{k}_{method}.csv")
    df.to_csv(out_csv, index=False)
    return out_csv


# ============================================================
# Plotting
# ============================================================

def plot_by_source(df: pd.DataFrame):
    """
    df columns: dataset, k, source, method, ndcg@10
    One PDF per dataset.
    """

    LINE_COLORS = {
        "RETRIEVER":   "#B0B0B0",
        "MonoBERT":    "#8DD3C7",
        "MonoT5_220M": "#FDB462",
        "MonoT5_3B":   "#80B1D3",
        "mMARCO_CE":   "#B3DE69",
        "GPT35":       "#FB8072",
        "GPT4oMini":   "#BC80BD",
    }
    DOT_COLORS = {
        "RETRIEVER":   "#424242",
        "MonoBERT":    "#00796B",
        "MonoT5_220M": "#E65100",
        "MonoT5_3B":   "#1565C0",
        "mMARCO_CE":   "#2E7D32",
        "GPT35":       "#B71C1C",
        "GPT4oMini":   "#6A1B9A",
    }

    DISPLAY_NAME = {
        "RETRIEVER": "Retriever (no rerank)",
        "GPT35": "GPT-3.5-turbo",
        "GPT4oMini": "GPT-4o-mini",
        "MonoT5_3B": "MonoT5-3B",
        "MonoT5_220M": "MonoT5-220M",
        "mMARCO_CE": "mMARCO-CE",
        "MonoBERT": "MonoBERT",
    }

    source_order = ["BM25", "RM3", "SPLADE"]
    x_map = {s: i for i, s in enumerate(source_order)}

    LEGEND_ORDER = [
        "RETRIEVER",
        "GPT4oMini",
        "MonoT5_3B",
        "MonoT5_220M",
        "mMARCO_CE",
        "GPT35",
        "MonoBERT",
    ]

    for dataset in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == dataset].copy()

        for k in sorted(sub["k"].unique()):
            subk = sub[sub["k"] == k].copy()
            subk["x"] = subk["source"].map(x_map)

            fig, ax = plt.subplots(figsize=(8, 5))

            present_methods = list(subk["method"].unique())
            methods = [m for m in LEGEND_ORDER if m in present_methods]
            methods += [m for m in sorted(present_methods) if m not in methods]

            for method in methods:
                mdf = subk[subk["method"] == method].sort_values("x")
                if mdf.empty:
                    continue

                ax.plot(
                    mdf["x"],
                    mdf["ndcg@10"],
                    label=DISPLAY_NAME.get(method, method),
                    linewidth=2.4,
                    marker="o",
                    markersize=6.5,
                    markerfacecolor=DOT_COLORS.get(method, "#424242"),
                    markeredgecolor=DOT_COLORS.get(method, "#424242"),
                    markeredgewidth=0.8,
                    color=LINE_COLORS.get(method, "#BDBDBD"),
                )

            present_sources = [s for s in source_order if s in set(subk["source"])]
            ax.set_xticks([x_map[s] for s in present_sources])
            ax.set_xticklabels(present_sources, fontsize=11)

            ax.set_xlabel("Input Source", fontsize=12)
            ax.set_ylabel("nDCG@10", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{dataset} | top-{k}", fontsize=12)

            ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
            fig.tight_layout(rect=[0, 0, 0.80, 1])

            out_path = os.path.join(RUN_PLOTS_ROOT, f"{dataset}_k{k}_ndcg_by_source.pdf")
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            print("[PLOT SAVED]", out_path)


# ============================================================
# Main
# ============================================================

def run_experiments():
    with open(RUN_LOG_TXT, "w") as runlog:
        log_line(runlog, f"RUN_TAG: {RUN_TAG}")
        log_line(runlog, f"DATASETS: {DATASETS}")
        log_line(runlog, f"SIZES: {SIZES}")
        log_line(runlog, f"INPUT_SOURCES: {INPUT_SOURCES}")
        log_line(runlog, f"METHODS: {METHODS}")
        log_line(runlog, f"TOP_K_EVAL (nDCG): {TOP_K_EVAL} | constant IDCG using max-grade per dataset")
        log_line(runlog, f"GAIN: 2^rel-1" if GAIN_POWER else "GAIN: linear rel")
        log_line(runlog, f"RBO_p: {RBO_P}")
        log_line(runlog, f"SIGNIFICANCE_ALPHA: {SIGNIFICANCE_ALPHA}")
        log_line(runlog, f"FORCE_RERUN: {FORCE_RERUN}")
        log_line(runlog, "-" * 90)

        if (("GPT35" in METHODS or "GPT4oMini" in METHODS) and not OPENAI_API_KEY):
            log_line(runlog, "[WARN] OPENAI_API_KEY is empty. GPT reranking will be skipped.")

        all_rows = []
        max_grade_per_dataset = {}

        for dataset in DATASETS:
            ds_start = time.time()
            log_line(runlog, "\n" + "=" * 90)
            log_line(runlog, f"[DATASET] {dataset}")
            log_line(runlog, "=" * 90)

            for k in SIZES:
                log_line(runlog, "\n" + "-" * 90)
                log_line(runlog, f"[SIZE] Top-{k}")
                log_line(runlog, "-" * 90)

                # For RBO between sources, we need the initial retriever docids per source
                retriever_docids_by_source: Dict[str, Dict[str, List[str]]] = {}
                # For significance, we need per-query ndcg by source per method
                per_query_ndcg_by_method: Dict[str, Dict[str, Dict[str, float]]] = {}

                for source in INPUT_SOURCES:
                    src_start = time.time()
                    log_line(runlog, f"\n[SOURCE] {source} | Top-{k}")

                    candidates, qrels, topics, searcher, topics_name_used, index_name_used = retrieve_candidates(dataset, source, k=k)
                    log_line(runlog, f"[INFO] topics={topics_name_used}")
                    log_line(runlog, f"[INFO] index={index_name_used}")

                    max_grade = max_grade_from_qrels(qrels)
                    idcg_const = idcg_constant(max_grade, TOP_K_EVAL)
                    max_grade_per_dataset[dataset] = max_grade
                    log_line(runlog, f"[INFO] max_grade={max_grade} | idcg_const@{TOP_K_EVAL}={idcg_const:.6f}")

                    # Use a separate FLAT searcher to fetch document text when source is SPLADE
                    text_searcher = None
                    if source.upper() == "SPLADE":
                        if dataset.lower() in ["dl19", "dl20"]:
                            text_searcher = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")
                        else:
                            text_searcher = LuceneSearcher.from_prebuilt_index(f"beir-v1.0.0-{dataset}.flat")

                    queries_map = normalize_topics_to_queries(topics)

                    def doc_fetcher(docid: str) -> str:
                        base = text_searcher if text_searcher is not None else searcher
                        txt = get_doc_text(base, str(docid))
                        return txt if isinstance(txt, str) else ""

                    # RETRIEVER baseline (no reranking)
                    retriever_docids = get_or_compute_method_docids(
                        dataset, source, k, "RETRIEVER",
                        candidates=candidates, topics=topics, searcher=searcher,
                        doc_fetcher=doc_fetcher, queries_map=queries_map
                    )
                    retriever_docids_by_source[source] = retriever_docids

                    retriever_run = run_from_docids(retriever_docids)
                    perq_retr = per_query_ndcg10_constant(retriever_run, qrels, idcg_const, TOP_K_EVAL)
                    retriever_ndcg = mean_of_dict(perq_retr)
                    log_line(runlog, f"[RESULT] {dataset} | k={k} | {source} | RETRIEVER | nDCG@{TOP_K_EVAL} = {retriever_ndcg:.6f}")
                    all_rows.append((dataset, k, source, "RETRIEVER", retriever_ndcg))

                    # log clean per-query files for retriever
                    input_docids_by_qid = {qid: [str(d) for d, _ in items[:k]] for qid, items in candidates.items()}
                    save_per_query_logs(dataset, source, k, "RETRIEVER", queries_map, input_docids_by_qid, retriever_docids, perq_retr)

                    per_query_ndcg_by_method.setdefault("RETRIEVER", {})[source] = perq_retr

                    # Rerankers
                    for method in METHODS:
                        if method in ["GPT35", "GPT4oMini"] and not OPENAI_API_KEY:
                            continue

                        docids_out = get_or_compute_method_docids(
                            dataset, source, k, method,
                            candidates=candidates, topics=topics, searcher=searcher,
                            doc_fetcher=doc_fetcher, queries_map=queries_map
                        )

                        run = run_from_docids(docids_out)
                        perq = per_query_ndcg10_constant(run, qrels, idcg_const, TOP_K_EVAL)
                        nd = mean_of_dict(perq)

                        log_line(runlog, f"[RESULT] {dataset} | k={k} | {source} | {method} | nDCG@{TOP_K_EVAL} = {nd:.6f}")
                        all_rows.append((dataset, k, source, method, nd))

                        save_per_query_logs(dataset, source, k, method, queries_map, input_docids_by_qid, docids_out, perq)
                        per_query_ndcg_by_method.setdefault(method, {})[source] = perq

                    src_time = time.time() - src_start
                    log_line(runlog, f"[TIME] {dataset} | k={k} | {source} finished in {src_time/60:.2f} min")

                    if source.upper() == "RM3" and searcher is not None and hasattr(searcher, "unset_rm3"):
                        try:
                            searcher.unset_rm3()
                        except Exception:
                            pass

                # ----- RBO between sources (initial rankings) -----
                if len(retriever_docids_by_source) >= 2:
                    rbo_csv, rbo_sum_csv = compute_and_save_rbo(dataset, k, retriever_docids_by_source)
                    log_line(runlog, f"[SAVED] RBO per-query -> {rbo_csv}")
                    log_line(runlog, f"[SAVED] RBO summary   -> {rbo_sum_csv}")

                # ----- Significance between sources (per method) -----
                for method, perq_by_source in per_query_ndcg_by_method.items():
                    if len(perq_by_source) >= 2:
                        sig_csv = compute_and_save_significance(dataset, k, method, perq_by_source)
                        log_line(runlog, f"[SAVED] Significance -> {sig_csv}")

                ds_time_k = time.time() - ds_start
                log_line(runlog, f"[TIME] {dataset} | k={k} finished in {ds_time_k/60:.2f} min")

            ds_time = time.time() - ds_start
            log_line(runlog, f"[TOTAL TIME] Dataset {dataset} finished in {ds_time/60:.2f} min")

        # Save combined outputs
        df_long = pd.DataFrame(all_rows, columns=["dataset", "k", "source", "method", "ndcg@10"])
        df_long.to_csv(RUN_RESULTS_CSV, index=False)
        log_line(runlog, f"\n[SAVED] Run long results  -> {RUN_RESULTS_CSV}")

        df_wide = df_long.pivot_table(index=["dataset", "k", "method"], columns="source", values="ndcg@10", aggfunc="mean").reset_index()
        df_wide.to_csv(RUN_RESULTS_WIDE_CSV, index=False)
        log_line(runlog, f"[SAVED] Run wide results  -> {RUN_RESULTS_WIDE_CSV}")

        atomic_write_json(RUN_MAX_GRADE_JSON, max_grade_per_dataset, indent=2)
        log_line(runlog, f"[SAVED] Max grade per ds  -> {RUN_MAX_GRADE_JSON}")

        plot_by_source(df_long)
        log_line(runlog, f"[DONE] Plots saved into: {RUN_PLOTS_ROOT}")


if __name__ == "__main__":
    run_experiments()