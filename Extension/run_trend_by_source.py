# ============================================================
# run_trend_by_source.py
# Compare candidate sources: BM25 vs RM3 vs SPLADE
# Supports: BEIR datasets + DL19/DL20 (MS MARCO passage)
# Fixed candidate size K (default 100)
# Saves into separate folders: *_by_source
# ALSO:
#   - logs per-(dataset,source,method): per-query ndcg@10 + input/output docids
#   - significance tests between sources (paired) per dataset & method
# ============================================================

import os
import json
import time
import math
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from pyserini.search import get_topics, get_qrels
from pyserini.search.lucene import LuceneSearcher, LuceneImpactSearcher

# ✅ SPLADE query encoder (required by LuceneImpactSearcher in your version)
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

from evaluate import eval_ndcg10

from rerank_supervised import MonoBERTReRanker, MonoT5ReRanker, rerank_with_progress as rerank_supervised
from rerank_unsupervised import CrossEncoderReRanker, rerank_with_progress as rerank_unsup
from rank_gpt import sliding_windows


# ============================================================
# CONFIG
# ============================================================

# DATASETS = [
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

DATASETS = [
#     "trec-covid",
    "dl19",
#     "dl20",
]

# Fixed candidate size for now
K = 100

# Candidate sources to compare
INPUT_SOURCES = ["BM25", "RM3", "SPLADE"]
# INPUT_SOURCES = ["SPLADE"]

# Rerankers to run on top of the candidate sets
METHODS = [
    "MonoBERT",
    "MonoT5_220M",
#     "MonoT5_3B",
#     "mMARCO_CE",
#     "GPT35",
#     "GPT4oMini",
]

# METHODS = [
#     "MonoBERT",
# ]

# ---------- OpenAI key ----------
OPENAI_API_KEY = ""
OPENAI_API_KEY = OPENAI_API_KEY.strip()

# RM3 parameters (common defaults)
RM3_FB_TERMS = 10
RM3_FB_DOCS = 10
RM3_ORIG_Q_WEIGHT = 0.5

# ============================================================
# OUTPUT ROOTS
# ============================================================

RESULTS_ROOT = "./results_by_source"
LOGS_ROOT    = "./logs_llm_rerank_sources"
PLOTS_ROOT   = "./plots_by_source"

os.makedirs(RESULTS_ROOT, exist_ok=True)
os.makedirs(LOGS_ROOT, exist_ok=True)
os.makedirs(PLOTS_ROOT, exist_ok=True)

RUN_LOG_TXT = os.path.join(RESULTS_ROOT, "run_log.txt")

# SPLADE query encoder name (unchanged from your latest version)
SPLADE_QUERY_ENCODER_NAME = "naver/splade-cocondenser-ensembledistil"


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def log_line(fp, msg: str):
    fp.write(msg.rstrip() + "\n")
    fp.flush()

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
    # Naming differs across setups; try common variants.
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

def splade_topics_candidates(dataset: str) -> List[str]:
    # Kept as-is (not used for SPLADE after earlier fix; you asked not to remove things)
    if is_msmarco_dl(dataset):
        return [topics_name_for(dataset)]
    base = f"beir-v1.0.0-{dataset}.test"
    return [
        f"{base}.splade-pp-ed",
        f"{base}.splade-pp",
        f"{base}.splade",
        topics_name_for(dataset),  # fallback
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

def run_from_candidates(candidates: Dict[str, List[Tuple[str, float]]], k: int) -> Dict[str, Dict[str, float]]:
    run = {}
    for qid, items in candidates.items():
        run[str(qid)] = {str(docid): float(score) for (docid, score) in items[:k]}
    return run

def make_scores_descending(order_docids: List[str]) -> Dict[str, float]:
    scores = {}
    n = len(order_docids)
    for i, did in enumerate(order_docids):
        scores[str(did)] = float(n - i)
    return scores

def compute_mean_ndcg10(run_dict: Dict[str, Dict[str, float]], qrels: Dict[str, Dict[str, int]]) -> float:
    return float(eval_ndcg10(run_dict, qrels))


# ============================================================
# NEW: Per-query NDCG@10 + logging utilities
# ============================================================

def compute_per_query_ndcg10(
    run_dict: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10,
) -> Dict[str, float]:
    """
    Returns {qid: ndcg@k}.
    Uses pytrec_eval if available.
    """
    try:
        import pytrec_eval
    except Exception as e:
        raise RuntimeError(
            "pytrec_eval is required for per-query ndcg logging/significance. "
            "Install it (pip/conda) and rerun. Original import error: "
            + repr(e)
        )

    # pytrec_eval expects qrels values as ints
    qrels_cast = {str(q): {str(d): int(r) for d, r in docs.items()} for q, docs in qrels.items()}
    run_cast   = {str(q): {str(d): float(s) for d, s in docs.items()} for q, docs in run_dict.items()}

    metric = f"ndcg_cut_{k}"
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_cast, {metric})
    res = evaluator.evaluate(run_cast)

    out: Dict[str, float] = {}
    for qid, md in res.items():
        out[str(qid)] = float(md.get(metric, 0.0))
    return out

def get_ranked_docids_from_run(run_for_qid: Dict[str, float]) -> List[str]:
    # sort docids by score desc
    return [d for d, _ in sorted(run_for_qid.items(), key=lambda x: x[1], reverse=True)]

def save_per_query_log_file(
    dataset: str,
    source: str,
    method: str,
    k: int,
    queries: Dict[str, str],
    input_docids_map: Dict[str, List[str]],
    output_docids_map: Dict[str, List[str]],
    per_query_ndcg: Dict[str, float],
):
    """
    Writes ONE file per (dataset, source, method):
      logs_llm_rerank_sources/k{k}/{dataset}/{source}/{method}/per_query.jsonl
    Each line: {qid, query, ndcg@10, input_docids, output_docids}
    Also writes a CSV version next to it.
    """
    out_dir = os.path.join(LOGS_ROOT, f"k{k}", dataset, source, method)
    ensure_dir(out_dir)

    jsonl_path = os.path.join(out_dir, "per_query.jsonl")
    csv_path   = os.path.join(out_dir, "per_query.csv")

    rows = []
    with open(jsonl_path, "w") as f:
        for qid in sorted(per_query_ndcg.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            obj = {
                "dataset": dataset,
                "input_source": source,
                "method": method,
                "k": k,
                "qid": str(qid),
                "query": queries.get(str(qid), ""),
                "ndcg@10": float(per_query_ndcg.get(str(qid), 0.0)),
                "input_docids": input_docids_map.get(str(qid), []),
                "output_docids": output_docids_map.get(str(qid), []),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            f.write(json.dumps(obj) + "\n")

            rows.append({
                "qid": str(qid),
                "ndcg@10": float(obj["ndcg@10"]),
                "query": obj["query"],
                "input_docids": " ".join(obj["input_docids"]),
                "output_docids": " ".join(obj["output_docids"]),
            })

    pd.DataFrame(rows).to_csv(csv_path, index=False)


# ============================================================
# NEW: significance tests between sources (paired)
# ============================================================

def paired_tests(a: List[float], b: List[float]) -> Dict[str, float]:
    """
    Paired tests on two aligned lists (same qids).
    Returns p-values for:
      - paired t-test
      - Wilcoxon signed-rank
      - sign test (binomial)
    If scipy missing, returns NaNs except sign test (we implement sign test ourselves).
    """
    # sign test (no ties)
    diffs = [x - y for x, y in zip(a, b)]
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg

    # two-sided exact binomial p-value
    def binom_two_sided_pvalue(k: int, n: int) -> float:
        if n == 0:
            return float("nan")
        # P(X <= min(k, n-k)) * 2 with p=0.5
        m = min(k, n - k)
        # compute sum_{i=0..m} C(n,i) / 2^n
        # use log-safe accumulation
        from math import comb
        s = 0.0
        for i in range(m + 1):
            s += comb(n, i)
        p = 2.0 * s / (2.0 ** n)
        return min(1.0, float(p))

    sign_p = binom_two_sided_pvalue(pos, n)

    # try scipy for t-test + wilcoxon
    t_p = float("nan")
    w_p = float("nan")
    try:
        from scipy import stats
        t_p = float(stats.ttest_rel(a, b, nan_policy="omit").pvalue)
        # Wilcoxon: requires at least one non-zero diff
        try:
            w_p = float(stats.wilcoxon(a, b, zero_method="wilcox", correction=False).pvalue)
        except Exception:
            w_p = float("nan")
    except Exception:
        pass

    return {"p_ttest": t_p, "p_wilcoxon": w_p, "p_signtest": sign_p, "n": int(n), "pos": int(pos), "neg": int(neg)}

def compute_significance_tables(
    dataset: str,
    k: int,
    per_query_store: Dict[Tuple[str, str], Dict[str, float]],
    sources_present: List[str],
    methods_present: List[str],
):
    """
    per_query_store key: (source, method) -> {qid: ndcg@10}
    Writes:
      results_by_source/significance_{dataset}.csv
    Comparisons are pairwise among sources, per method.
    """
    pairs = []
    order = ["BM25", "RM3", "SPLADE"]
    sp = [s for s in order if s in sources_present]
    # pairwise
    comps = []
    for i in range(len(sp)):
        for j in range(i + 1, len(sp)):
            comps.append((sp[i], sp[j]))

    rows = []
    for method in methods_present:
        for s1, s2 in comps:
            a_map = per_query_store.get((s1, method), {})
            b_map = per_query_store.get((s2, method), {})
            # align on intersection of qids
            qids = sorted(set(a_map.keys()) & set(b_map.keys()), key=lambda x: int(x) if str(x).isdigit() else str(x))
            a = [a_map[q] for q in qids]
            b = [b_map[q] for q in qids]
            res = paired_tests(a, b)
            rows.append({
                "dataset": dataset,
                "k": k,
                "method": method,
                "source_A": s1,
                "source_B": s2,
                "mean_A": float(sum(a)/len(a)) if len(a) else float("nan"),
                "mean_B": float(sum(b)/len(b)) if len(b) else float("nan"),
                "p_ttest": res["p_ttest"],
                "p_wilcoxon": res["p_wilcoxon"],
                "p_signtest": res["p_signtest"],
                "n_effective": res["n"],
                "pos": res["pos"],
                "neg": res["neg"],
            })

    out_path = os.path.join(RESULTS_ROOT, f"significance_{dataset}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)


# ============================================================
# Candidate retrieval by source
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

    if source in ["BM25", "RM3"]:
        searcher, index_name_used = open_prebuilt_lucene_index(bm25_index_candidates(dataset))

        if source == "RM3":
            searcher.set_rm3(RM3_FB_TERMS, RM3_FB_DOCS, RM3_ORIG_Q_WEIGHT)

        topics_name_used = topics_name_for(dataset)
        topics = get_topics(topics_name_used)
        qrels = get_qrels(topics_name_used)

    elif source == "SPLADE":
        query_encoder = SpladeQueryEncoder(SPLADE_QUERY_ENCODER_NAME)
        searcher, index_name_used = open_prebuilt_impact_index(splade_index_candidates(dataset), query_encoder)

        # Always use NORMAL topics/qrels; SpladeQueryEncoder handles encoding
        topics_name_used = topics_name_for(dataset)
        topics = get_topics(topics_name_used)
        qrels = get_qrels(topics_name_used)

    else:
        raise ValueError(f"Unknown input source: {source}")

    queries = normalize_topics_to_queries(topics)

    # =========================================================
    # Restrict queries to qrels qids (fix 200 vs 50)
    # =========================================================
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

    return candidates, qrels, topics, searcher, topics_name_used, index_name_used


# ============================================================
# Rerankers
# ============================================================

def rerank_supervised_models(method: str, candidates, queries, doc_fetcher, batch_size=8) -> Dict[str, Dict[str, float]]:
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

    run = {}
    for qid, pairs in rr.items():
        run[str(qid)] = {str(docid): float(score) for docid, score in pairs}
    return run

def rerank_mmarco_ce(candidates, queries, doc_fetcher, batch_size=16) -> Dict[str, Dict[str, float]]:
    scorer = CrossEncoderReRanker(MMARCO_CE_CKPT)
    rr = rerank_unsup("mMARCO-CE", scorer, candidates, queries, doc_fetcher,
                      batch_size=batch_size, colour="yellow")

    run = {}
    for qid, pairs in rr.items():
        run[str(qid)] = {str(docid): float(score) for docid, score in pairs}
    return run

def rerank_rankgpt_llm(dataset: str, candidates, topics, searcher, k: int,
                       model_name: str, openai_key: str, source: str) -> Dict[str, Dict[str, float]]:
    run = {}

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
        out_docids = [h["docid"] for h in reranked_hits]
        run[qid_str] = make_scores_descending(out_docids)

        # keep your existing per-qid json log for LLM
        out_dir = os.path.join(LOGS_ROOT, f"k{k}", dataset, source, model_name)
        ensure_dir(out_dir)
        out_path = os.path.join(out_dir, f"qid_{qid_str}.json")
        obj = {
            "dataset": dataset,
            "input_source": source,
            "method": model_name,
            "k": k,
            "qid": qid_str,
            "query": query,
            "input_docids": [str(d) for (d, _) in items],
            "output_docids": out_docids,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(out_path, "w") as f:
            json.dump(obj, f, indent=2)

    return run


# ============================================================
# Plotting
# ============================================================

def plot_by_source(df: pd.DataFrame):
    """
    df columns: dataset, source, method, ndcg@10
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
        sub["x"] = sub["source"].map(x_map)

        fig, ax = plt.subplots(figsize=(8, 5))

        present_methods = list(sub["method"].unique())
        methods = [m for m in LEGEND_ORDER if m in present_methods]
        methods += [m for m in sorted(present_methods) if m not in methods]

        for method in methods:
            mdf = sub[sub["method"] == method].sort_values("x")
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

        present_sources = [s for s in source_order if s in set(sub["source"])]
        ax.set_xticks([x_map[s] for s in present_sources])
        ax.set_xticklabels(present_sources, fontsize=11)

        ax.set_xlabel("Input Source", fontsize=12)
        ax.set_ylabel("NDCG@10", fontsize=12)
        ax.grid(True, alpha=0.3)

        ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
        fig.tight_layout(rect=[0, 0, 0.80, 1])

        out_path = os.path.join(PLOTS_ROOT, f"{dataset}_ndcg_by_source.pdf")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print("[PLOT SAVED]", out_path)


# ============================================================
# Main
# ============================================================

def run_experiments():
    with open(RUN_LOG_TXT, "w") as runlog:
        log_line(runlog, f"DATASETS: {DATASETS}")
        log_line(runlog, f"K: {K}")
        log_line(runlog, f"INPUT_SOURCES: {INPUT_SOURCES}")
        log_line(runlog, f"METHODS: {METHODS}")
        log_line(runlog, "-" * 90)

        if not OPENAI_API_KEY and ("GPT35" in METHODS or "GPT4oMini" in METHODS):
            log_line(runlog, "[WARN] OPENAI_API_KEY is empty. GPT reranking will be skipped.")

        all_rows = []

        for dataset in DATASETS:
            ds_start = time.time()
            log_line(runlog, "\n" + "=" * 90)
            log_line(runlog, f"[DATASET] {dataset}")
            log_line(runlog, "=" * 90)

            # store per-query ndcg maps for significance: (source, method) -> {qid: ndcg}
            per_query_store: Dict[Tuple[str, str], Dict[str, float]] = {}

            for source in INPUT_SOURCES:
                src_start = time.time()
                log_line(runlog, f"\n[SOURCE] {source} | Top-{K}")

                candidates, qrels, topics, searcher, topics_name_used, index_name_used = retrieve_candidates(dataset, source, k=K)

                # Use a separate FLAT searcher to fetch doc text when source is SPLADE
                text_searcher = None
                if source.upper() == "SPLADE":
                    if dataset.lower() in ["dl19", "dl20"]:
                        text_searcher = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")
                    else:
                        text_searcher = LuceneSearcher.from_prebuilt_index(f"beir-v1.0.0-{dataset}.flat")

                log_line(runlog, f"[INFO] topics={topics_name_used}")
                log_line(runlog, f"[INFO] index={index_name_used}")

                queries = normalize_topics_to_queries(topics)

                def doc_fetcher(docid: str) -> str:
                    base = text_searcher if text_searcher is not None else searcher
                    txt = get_doc_text(base, str(docid))
                    return txt if isinstance(txt, str) else ""

                # -------------------------
                # Retriever baseline
                # -------------------------
                retrieval_run = run_from_candidates(candidates, k=K)
                retrieval_ndcg = compute_mean_ndcg10(retrieval_run, qrels)
                log_line(runlog, f"[RESULT] {dataset} | {source} | RETRIEVER | nDCG@10 = {retrieval_ndcg:.4f}")
                all_rows.append((dataset, source, "RETRIEVER", retrieval_ndcg))

                # per-query ndcg + log file for RETRIEVER
                per_q = compute_per_query_ndcg10(retrieval_run, qrels, k=10)
                per_query_store[(source, "RETRIEVER")] = per_q

                input_docids_map = {str(qid): [d for (d, _) in items[:K]] for qid, items in candidates.items()}
                output_docids_map = {qid: get_ranked_docids_from_run(retrieval_run[qid]) for qid in retrieval_run.keys()}

                save_per_query_log_file(
                    dataset=dataset,
                    source=source,
                    method="RETRIEVER",
                    k=K,
                    queries=queries,
                    input_docids_map=input_docids_map,
                    output_docids_map=output_docids_map,
                    per_query_ndcg=per_q,
                )

                # -------------------------
                # Supervised rerankers
                # -------------------------
                if "MonoBERT" in METHODS:
                    run = rerank_supervised_models("MonoBERT", candidates, queries, doc_fetcher, batch_size=8)
                    nd = compute_mean_ndcg10(run, qrels)
                    log_line(runlog, f"[RESULT] {dataset} | {source} | MonoBERT | nDCG@10 = {nd:.4f}")
                    all_rows.append((dataset, source, "MonoBERT", nd))

                    per_q = compute_per_query_ndcg10(run, qrels, k=10)
                    per_query_store[(source, "MonoBERT")] = per_q

                    output_docids_map = {qid: get_ranked_docids_from_run(run[qid]) for qid in run.keys()}
                    save_per_query_log_file(
                        dataset=dataset,
                        source=source,
                        method="MonoBERT",
                        k=K,
                        queries=queries,
                        input_docids_map=input_docids_map,
                        output_docids_map=output_docids_map,
                        per_query_ndcg=per_q,
                    )

                if "MonoT5_220M" in METHODS:
                    run = rerank_supervised_models("MonoT5_220M", candidates, queries, doc_fetcher, batch_size=8)
                    nd = compute_mean_ndcg10(run, qrels)
                    log_line(runlog, f"[RESULT] {dataset} | {source} | MonoT5_220M | nDCG@10 = {nd:.4f}")
                    all_rows.append((dataset, source, "MonoT5_220M", nd))

                    per_q = compute_per_query_ndcg10(run, qrels, k=10)
                    per_query_store[(source, "MonoT5_220M")] = per_q

                    output_docids_map = {qid: get_ranked_docids_from_run(run[qid]) for qid in run.keys()}
                    save_per_query_log_file(
                        dataset=dataset,
                        source=source,
                        method="MonoT5_220M",
                        k=K,
                        queries=queries,
                        input_docids_map=input_docids_map,
                        output_docids_map=output_docids_map,
                        per_query_ndcg=per_q,
                    )

                if "MonoT5_3B" in METHODS:
                    run = rerank_supervised_models("MonoT5_3B", candidates, queries, doc_fetcher, batch_size=4)
                    nd = compute_mean_ndcg10(run, qrels)
                    log_line(runlog, f"[RESULT] {dataset} | {source} | MonoT5_3B | nDCG@10 = {nd:.4f}")
                    all_rows.append((dataset, source, "MonoT5_3B", nd))

                    per_q = compute_per_query_ndcg10(run, qrels, k=10)
                    per_query_store[(source, "MonoT5_3B")] = per_q

                    output_docids_map = {qid: get_ranked_docids_from_run(run[qid]) for qid in run.keys()}
                    save_per_query_log_file(
                        dataset=dataset,
                        source=source,
                        method="MonoT5_3B",
                        k=K,
                        queries=queries,
                        input_docids_map=input_docids_map,
                        output_docids_map=output_docids_map,
                        per_query_ndcg=per_q,
                    )

                # -------------------------
                # Unsupervised CE
                # -------------------------
                if "mMARCO_CE" in METHODS:
                    run = rerank_mmarco_ce(candidates, queries, doc_fetcher, batch_size=16)
                    nd = compute_mean_ndcg10(run, qrels)
                    log_line(runlog, f"[RESULT] {dataset} | {source} | mMARCO-CE | nDCG@10 = {nd:.4f}")
                    all_rows.append((dataset, source, "mMARCO_CE", nd))

                    per_q = compute_per_query_ndcg10(run, qrels, k=10)
                    per_query_store[(source, "mMARCO_CE")] = per_q

                    output_docids_map = {qid: get_ranked_docids_from_run(run[qid]) for qid in run.keys()}
                    save_per_query_log_file(
                        dataset=dataset,
                        source=source,
                        method="mMARCO_CE",
                        k=K,
                        queries=queries,
                        input_docids_map=input_docids_map,
                        output_docids_map=output_docids_map,
                        per_query_ndcg=per_q,
                    )

                # -------------------------
                # LLM rerankers
                # -------------------------
                if OPENAI_API_KEY:
                    if "GPT35" in METHODS:
                        run = rerank_rankgpt_llm(dataset, candidates, topics, searcher, K, "gpt-3.5-turbo", OPENAI_API_KEY, source)
                        nd = compute_mean_ndcg10(run, qrels)
                        log_line(runlog, f"[RESULT] {dataset} | {source} | GPT-3.5-turbo | nDCG@10 = {nd:.4f}")
                        all_rows.append((dataset, source, "GPT35", nd))

                        per_q = compute_per_query_ndcg10(run, qrels, k=10)
                        per_query_store[(source, "GPT35")] = per_q

                        output_docids_map = {qid: get_ranked_docids_from_run(run[qid]) for qid in run.keys()}
                        save_per_query_log_file(
                            dataset=dataset,
                            source=source,
                            method="GPT35",
                            k=K,
                            queries=queries,
                            input_docids_map=input_docids_map,
                            output_docids_map=output_docids_map,
                            per_query_ndcg=per_q,
                        )

                    if "GPT4oMini" in METHODS:
                        run = rerank_rankgpt_llm(dataset, candidates, topics, searcher, K, "gpt-4o-mini", OPENAI_API_KEY, source)
                        nd = compute_mean_ndcg10(run, qrels)
                        log_line(runlog, f"[RESULT] {dataset} | {source} | GPT-4o-mini | nDCG@10 = {nd:.4f}")
                        all_rows.append((dataset, source, "GPT4oMini", nd))

                        per_q = compute_per_query_ndcg10(run, qrels, k=10)
                        per_query_store[(source, "GPT4oMini")] = per_q

                        output_docids_map = {qid: get_ranked_docids_from_run(run[qid]) for qid in run.keys()}
                        save_per_query_log_file(
                            dataset=dataset,
                            source=source,
                            method="GPT4oMini",
                            k=K,
                            queries=queries,
                            input_docids_map=input_docids_map,
                            output_docids_map=output_docids_map,
                            per_query_ndcg=per_q,
                        )

                src_time = time.time() - src_start
                log_line(runlog, f"[TIME] {dataset} | {source} finished in {src_time/60:.2f} min")

                if source.upper() == "RM3" and hasattr(searcher, "unset_rm3"):
                    try:
                        searcher.unset_rm3()
                    except Exception:
                        pass

            # Save outputs per dataset (mean results)
            df_dataset = pd.DataFrame(
                [r for r in all_rows if r[0] == dataset],
                columns=["dataset", "source", "method", "ndcg@10"]
            )

            out_csv   = os.path.join(RESULTS_ROOT, f"ndcg_by_source_all_{dataset}.csv")
            pivot_csv = os.path.join(RESULTS_ROOT, f"ndcg_by_source_pivot_{dataset}.csv")
            summary   = os.path.join(RESULTS_ROOT, f"summary_{dataset}.txt")

            df_dataset.to_csv(out_csv, index=False)
            log_line(runlog, f"[SAVED] Results CSV -> {out_csv}")

            pivot = df_dataset.pivot_table(index=["dataset", "method"], columns="source", values="ndcg@10", aggfunc="mean")
            pivot.to_csv(pivot_csv)
            log_line(runlog, f"[SAVED] Pivot CSV -> {pivot_csv}")

            with open(summary, "w") as sf:
                sf.write(f"DATASETS: [{dataset}]\n")
                sf.write(f"K: {K}\n")
                sf.write(f"INPUT_SOURCES: {INPUT_SOURCES}\n")
                sf.write(f"METHODS: {METHODS}\n\n")
                sf.write("=== nDCG@10 Pivot ===\n\n")
                sf.write(pivot.to_string())
                sf.write("\n")
            log_line(runlog, f"[SAVED] Summary TXT -> {summary}")

            plot_by_source(df_dataset)
            log_line(runlog, f"[DONE] Plot saved into: {PLOTS_ROOT}")

            # -------------------------
            # Significance tests between sources (paired), per method
            # Uses per-query NDCG@10 already computed and aligned by qid intersection.
            # -------------------------
            sources_present = sorted(set([s for (s, m) in per_query_store.keys()]))
            methods_present = sorted(set([m for (s, m) in per_query_store.keys()]))

            compute_significance_tables(
                dataset=dataset,
                k=K,
                per_query_store=per_query_store,
                sources_present=sources_present,
                methods_present=methods_present,
            )
            log_line(runlog, f"[SAVED] Significance CSV -> {os.path.join(RESULTS_ROOT, f'significance_{dataset}.csv')}")

            ds_time = time.time() - ds_start
            log_line(runlog, f"[TOTAL TIME] Dataset {dataset} finished in {ds_time/60:.2f} min")


if __name__ == "__main__":
    run_experiments()
