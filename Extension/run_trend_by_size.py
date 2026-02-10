OPENAI_API_KEY = ""



# ============================================================
# run_trend_by_size.py
# ============================================================

import os
import json
import time
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------------------
#  modules
# ------------------------------
from config import (
    MONOBERT_CKPT,
    MONOT5_BASE_CKPT,
    MONOT5_3B_CKPT,
    MMARCO_CE_CKPT,
)

from retrieve_bm25 import retrieve_topk
from evaluate import eval_ndcg10

from rerank_supervised import MonoBERTReRanker, MonoT5ReRanker, rerank_with_progress as rerank_supervised
from rerank_unsupervised import CrossEncoderReRanker, rerank_with_progress as rerank_unsup

from rank_gpt import sliding_windows


# ============================================================
# CONFIG: datasets, methods, sizes
# ============================================================

# ✅ Full dataset list (commented)
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
    "dl19",
]

SIZES = [10, 20, 50, 100]   # later: [10,20,50,100,200]


# ✅ Full method list (commented)
# METHODS = [
#     "BM25",
#     "MonoBERT",
#     "MonoT5_220M",
#     "MonoT5_3B",
#     "mMARCO_CE",
#     "GPT35",
#     "GPT4oMini",
# ]

METHODS = [
    "BM25",
    "MonoBERT",
    "MonoT5_220M",
    "MonoT5_3B",
    "mMARCO_CE",
#     "GPT35",
#     "GPT4oMini",
]



# ============================================================
# OUTPUT ROOTS (fixed filenames, NO run_id)
# ============================================================

RESULTS_ROOT = "./results_by_size"
LOGS_ROOT    = "./logs_llm_rerank_sizes"
PLOTS_ROOT   = "./plots_by_size"

os.makedirs(RESULTS_ROOT, exist_ok=True)
os.makedirs(LOGS_ROOT, exist_ok=True)
os.makedirs(PLOTS_ROOT, exist_ok=True)

RUN_LOG_TXT  = os.path.join(RESULTS_ROOT, "run_log.txt")


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def log_line(fp, msg: str):
    fp.write(msg.rstrip() + "\n")
    fp.flush()

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

def save_qid_log(root: str, k: int, dataset: str, method: str, qid: str, query: str,
                 input_docids: List[str], output_docids: List[str],
                 input_scores=None, output_scores=None):
    out_dir = os.path.join(root, f"k{k}", dataset, method)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"qid_{qid}.json")

    obj = {
        "dataset": dataset,
        "method": method,
        "k": k,
        "qid": str(qid),
        "query": query,
        "input_docids": list(map(str, input_docids)),
        "output_docids": list(map(str, output_docids)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if input_scores is not None:
        obj["input_scores"] = input_scores
    if output_scores is not None:
        obj["output_scores"] = output_scores

    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2)

def compute_mean_ndcg10(run_dict: Dict[str, Dict[str, float]], qrels: Dict[str, Dict[str, int]]) -> float:
    return float(eval_ndcg10(run_dict, qrels))

def make_scores_descending(order_docids: List[str]) -> Dict[str, float]:
    scores = {}
    n = len(order_docids)
    for i, did in enumerate(order_docids):
        scores[str(did)] = float(n - i)
    return scores


# ============================================================
# NEW: per-query NDCG + per-qid logs + significance
# ============================================================

import math
from itertools import combinations

try:
    from scipy.stats import wilcoxon, friedmanchisquare
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

def _dcg_at_k(rels: List[float], k: int = 10) -> float:
    s = 0.0
    for i, rel in enumerate(rels[:k]):
        s += float(rel) / math.log2(i + 2)
    return s

def _ndcg_at_10_for_qid(qid: str, ranked_docids: List[str], qrels: Dict[str, Dict[str, int]]) -> float:
    qid = str(qid)
    qrel_q = qrels.get(qid, {}) or {}

    rels = [float(qrel_q.get(str(did), 0)) for did in ranked_docids[:10]]
    dcg = _dcg_at_k(rels, k=10)

    ideal_rels = sorted([float(v) for v in qrel_q.values()], reverse=True)
    idcg = _dcg_at_k(ideal_rels, k=10)

    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)

def compute_per_query_ndcg10_from_docids(
    per_query_docids: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]]
) -> Dict[str, float]:
    out = {}
    for qid, docids in per_query_docids.items():
        out[str(qid)] = _ndcg_at_10_for_qid(str(qid), [str(d) for d in docids], qrels)
    return out

def save_qid_log_with_ndcg(
    root: str,
    k: int,
    dataset: str,
    method: str,
    qid: str,
    query: str,
    input_docids: List[str],
    output_docids: List[str],
    ndcg10: float,
    input_scores=None,
    output_scores=None,
):
    out_dir = os.path.join(root, f"k{k}", dataset, method)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"qid_{qid}.json")

    obj = {
        "dataset": dataset,
        "method": method,
        "k": k,
        "qid": str(qid),
        "query": query,
        "input_docids": list(map(str, input_docids)),
        "output_docids": list(map(str, output_docids)),
        "ndcg@10": float(ndcg10),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if input_scores is not None:
        obj["input_scores"] = input_scores
    if output_scores is not None:
        obj["output_scores"] = output_scores

    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2)

def save_per_query_result_log(
    root: str,
    dataset: str,
    method: str,
    k: int,
    per_q_ndcg: Dict[str, float],
    mean_ndcg: float,
    note: str = ""
):
    out_dir = os.path.join(root, "per_query_ndcg", dataset, method)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"k{k}.json")

    obj = {
        "dataset": dataset,
        "method": method,
        "k": k,
        "mean_ndcg@10": float(mean_ndcg),
        "per_query_ndcg@10": {str(q): float(v) for q, v in per_q_ndcg.items()},
        "note": note,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2)

def _paired_wilcoxon(x: List[float], y: List[float]):
    if not _HAVE_SCIPY:
        return None, 0, None

    pairs = [(a, b) for a, b in zip(x, y)
             if a is not None and b is not None and not (math.isnan(a) or math.isnan(b))]
    if not pairs:
        return None, 0, None

    xx = [a for a, _ in pairs]
    yy = [b for _, b in pairs]

    diffs = [yy[i] - xx[i] for i in range(len(xx))]
    if all(abs(d) < 1e-12 for d in diffs):
        return 1.0, len(xx), 0.0

    try:
        stat = wilcoxon(xx, yy, zero_method="wilcox", alternative="two-sided", mode="auto")
        p = float(stat.pvalue)
    except Exception:
        p = None

    diffs_sorted = sorted(diffs)
    mid = len(diffs_sorted) // 2
    med = diffs_sorted[mid] if len(diffs_sorted) % 2 == 1 else 0.5 * (diffs_sorted[mid - 1] + diffs_sorted[mid])
    return p, len(xx), float(med)

def _friedman_test(matrix: List[List[float]]):
    if not _HAVE_SCIPY:
        return None
    try:
        stat = friedmanchisquare(*matrix)
        return float(stat.pvalue)
    except Exception:
        return None

def run_significance_by_size(
    per_query_by_k: Dict[int, Dict[str, float]],
    sizes: List[int]
) -> Dict:
    common_qids = None
    for k in sizes:
        qset = set(per_query_by_k.get(k, {}).keys())
        common_qids = qset if common_qids is None else (common_qids & qset)
    common_qids = sorted(common_qids or [])

    vec_by_k = {}
    for k in sizes:
        vec_by_k[k] = [float(per_query_by_k[k][qid]) for qid in common_qids]

    out = {
        "n_common_qids": len(common_qids),
        "sizes": list(sizes),
        "pairwise_wilcoxon": [],
        "friedman_p": None,
    }

    if len(common_qids) == 0:
        return out

    for a, b in combinations(sizes, 2):
        p, n_used, med_delta = _paired_wilcoxon(vec_by_k[a], vec_by_k[b])
        out["pairwise_wilcoxon"].append({
            "a": int(a),
            "b": int(b),
            "p_value": p,
            "n_used": int(n_used),
            "median_delta(b-a)": med_delta
        })

    matrix = [vec_by_k[k] for k in sizes]
    out["friedman_p"] = _friedman_test(matrix)
    return out

def save_significance_result(
    root: str,
    dataset: str,
    method: str,
    sig_obj: Dict
):
    out_dir = os.path.join(root, "significance_by_size", dataset)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{method}_significance.json")
    with open(out_path, "w") as f:
        json.dump(sig_obj, f, indent=2)

def _build_per_query_io_docids_from_run(
    bm25_res: Dict,
    run_dict: Dict[str, Dict[str, float]],
    k: int
):
    """
    Returns:
      input_docids_by_qid:  {qid: [docid,...]} from BM25 top-k
      output_docids_by_qid: {qid: [docid,...]} from run_dict sorted by score desc
    """
    def safe_get_items(qid_str: str):
        if qid_str in bm25_res:
            return bm25_res[qid_str]
        try:
            qi = int(qid_str)
            if qi in bm25_res:
                return bm25_res[qi]
        except Exception:
            pass
        return []

    input_docids_by_qid = {}
    output_docids_by_qid = {}

    for qid in run_dict.keys():
        qid_str = str(qid)
        items = safe_get_items(qid_str)[:k]
        input_docids_by_qid[qid_str] = [str(d) for (d, _) in items]

        ranked = sorted(run_dict[qid_str].items(), key=lambda x: x[1], reverse=True)
        output_docids_by_qid[qid_str] = [str(docid) for docid, _ in ranked][:k]

    return input_docids_by_qid, output_docids_by_qid


# ============================================================
# Rerankers
# ============================================================

def rerank_bm25(bm25_res: Dict[str, List[Tuple[str, float]]], k: int) -> Dict[str, Dict[str, float]]:
    run = {}
    for qid, items in bm25_res.items():
        run[str(qid)] = {str(docid): float(score) for (docid, score) in items[:k]}
    return run

def rerank_supervised_models(method: str, bm25_res, queries, doc_fetcher, batch_size=8) -> Dict[str, Dict[str, float]]:
    if method == "MonoBERT":
        scorer = MonoBERTReRanker(MONOBERT_CKPT, max_len=512)
        rr = rerank_supervised("monoBERT", scorer, bm25_res, queries, doc_fetcher,
                               batch_size=batch_size, colour="blue")

    elif method == "MonoT5_220M":
        scorer = MonoT5ReRanker(MONOT5_BASE_CKPT)
        rr = rerank_supervised("monoT5-base", scorer, bm25_res, queries, doc_fetcher,
                               batch_size=batch_size, colour="blue")

    elif method == "MonoT5_3B":
        scorer = MonoT5ReRanker(MONOT5_3B_CKPT)
        rr = rerank_supervised("monoT5-3B", scorer, bm25_res, queries, doc_fetcher,
                               batch_size=max(1, batch_size // 2), colour="blue")

    else:
        raise ValueError(f"Unknown supervised method: {method}")

    run = {}
    for qid, pairs in rr.items():
        run[str(qid)] = {str(docid): float(score) for docid, score in pairs}
    return run

def rerank_mmarco_ce(bm25_res, queries, doc_fetcher, batch_size=16) -> Dict[str, Dict[str, float]]:
    scorer = CrossEncoderReRanker(MMARCO_CE_CKPT)
    rr = rerank_unsup("mMARCO-CE", scorer, bm25_res, queries, doc_fetcher,
                      batch_size=batch_size, colour="yellow")

    run = {}
    for qid, pairs in rr.items():
        run[str(qid)] = {str(docid): float(score) for docid, score in pairs}
    return run

def rerank_rankgpt_llm(dataset: str, bm25_res, topics, searcher, k: int,
                       model_name: str, openai_key: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[str]]]:
    """
    Returns:
      run: {qid: {docid: score}}
      output_docids_by_qid: {qid: [docid,...]}  (rank order after LLM)
    """
    run = {}
    output_docids_by_qid = {}

    def safe_get_items(qid_str: str):
        """bm25_res keys might be int (0,1,2,...) or str ('0','1',...)"""
        if qid_str in bm25_res:
            return bm25_res[qid_str]
        try:
            qid_int = int(qid_str)
            if qid_int in bm25_res:
                return bm25_res[qid_int]
        except Exception:
            pass
        return []

    def qtext_of(qid_str: str):
        """topics keys might be int or str as well"""
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

    qids = list(bm25_res.keys())
    try:
        qids = sorted(qids, key=lambda x: int(x))
    except Exception:
        qids = sorted(map(str, qids))

    iterator = tqdm(qids, desc=f"{dataset}-{model_name}-k{k}", ncols=110)

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
        output_docids_by_qid[qid_str] = out_docids

    return run, output_docids_by_qid



# ============================================================
# Plotting
# ============================================================

def plot_trends(
    df: pd.DataFrame,
    title_size: int = 14,   # kept for compatibility (title not shown)
    label_size: int = 12,
    tick_size: int = 11,
    legend_size: int = 10,
    legend_loc: str = "upper left",
    legend_outside: bool = True,
):
    """
    Styled plot:
      - dark dots + light lines (fixed per method)
      - ax.grid(True, alpha=0.3)
      - custom legend order + pretty legend labels
      - NO title
      - y-axis label: NDCG
      - x-axis label: Input Size
    """

    # =========================
    # Fixed, highly distinct palette (per method)
    # =========================
    LINE_COLORS = {
        "BM25":        "#B0B0B0",
        "MonoBERT":    "#8DD3C7",
        "MonoT5_220M": "#FDB462",
        "MonoT5_3B":   "#80B1D3",
        "mMARCO_CE":   "#B3DE69",
        "GPT35":       "#FB8072",
        "GPT4oMini":   "#BC80BD",
    }

    DOT_COLORS = {
        "BM25":        "#424242",
        "MonoBERT":    "#00796B",
        "MonoT5_220M": "#E65100",
        "MonoT5_3B":   "#1565C0",
        "mMARCO_CE":   "#2E7D32",
        "GPT35":       "#B71C1C",
        "GPT4oMini":   "#6A1B9A",
    }

    # =========================
    # Legend order + display names
    # =========================
    LEGEND_ORDER = [
        "GPT4oMini",
        "MonoT5_3B",
        "MonoT5_220M",
        "mMARCO_CE",
        "GPT35",
        "MonoBERT",
        "BM25",
    ]

    DISPLAY_NAME = {
        "GPT35": "GPT-3.5-turbo",
        "GPT4oMini": "GPT-4o-mini",
        "MonoT5_3B": "MonoT5-3B",
        "MonoT5_220M": "MonoT5-220M",
        "mMARCO_CE": "mMARCO-CE",
        "MonoBERT": "MonoBERT",
        "BM25": "BM25",
    }

    def pretty_label(method: str) -> str:
        # Use explicit mapping first; otherwise replace underscores with hyphens
        return DISPLAY_NAME.get(method, method.replace("_", "-"))

    for dataset in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == dataset].copy()
        present = set(sub["method"].unique())

        # enforce your legend order, but only for methods present in df
        methods = [m for m in LEGEND_ORDER if m in present]
        # include any unexpected methods at the end (still plotted)
        methods += [m for m in sorted(present) if m not in methods]

        fig, ax = plt.subplots(figsize=(8, 5))

        for method in methods:
            mdf = sub[sub["method"] == method].sort_values("k")
            if mdf.empty:
                continue

            line_color   = LINE_COLORS.get(method, "#BDBDBD")
            marker_color = DOT_COLORS.get(method, "#424242")

            ax.plot(
                mdf["k"],
                mdf["ndcg@10"],
                label=pretty_label(method),
                color=line_color,
                linewidth=2.4,
                marker="o",
                markersize=6.5,
                markerfacecolor=marker_color,
                markeredgecolor=marker_color,
                markeredgewidth=0.8,
            )

        # NO title
        ax.set_xlabel("Input Size", fontsize=label_size)
        ax.set_ylabel("NDCG@10", fontsize=label_size)

        ax.set_xticks(sorted(sub["k"].unique()))
        ax.tick_params(axis="both", labelsize=tick_size)
        ax.grid(True, alpha=0.3)

        # Legend placement
        if legend_outside:
            ax.legend(
                fontsize=legend_size,
                loc=legend_loc,
                bbox_to_anchor=(1.02, 1),
                borderaxespad=0.0,
                frameon=True
            )
            fig.tight_layout(rect=[0, 0, 0.80, 1])
        else:
            ax.legend(fontsize=legend_size, loc=legend_loc, frameon=True)
            fig.tight_layout()

        out_path = os.path.join(PLOTS_ROOT, f"{dataset}_ndcg_trend.pdf")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

        print("[PLOT SAVED]", out_path)



def plot_datasets_from_csv(
    datasets: List[str],
    results_root: str = RESULTS_ROOT,
    title_size: int = 14,
    label_size: int = 12,
    tick_size: int = 11,
    legend_size: int = 10,
    legend_loc: str = "upper left",
    legend_outside: bool = True,
):
    """
    For each dataset in `datasets`:
      - reads:  results_by_size/ndcg_trend_all_{dataset}.csv
      - plots:  plots_by_size/{dataset}_ndcg_trend.pdf  (via plot_trends)
    """

    for dataset in datasets:
        csv_path = os.path.join(results_root, f"ndcg_trend_all_{dataset}.csv")

        if not os.path.exists(csv_path):
            print(f"[SKIP] Missing CSV for dataset='{dataset}': {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # Optional: keep only this dataset (in case CSV contains more)
        df = df[df["dataset"] == dataset].copy()
        if df.empty:
            print(f"[SKIP] CSV has no rows for dataset='{dataset}': {csv_path}")
            continue

        # Reuse your existing plot function (it already saves PDF per dataset)
        plot_trends(
            df,
            title_size=title_size,
            label_size=label_size,
            tick_size=tick_size,
            legend_size=legend_size,
            legend_loc=legend_loc,
            legend_outside=legend_outside,
        )

        print(f"[DONE] Plotted dataset='{dataset}' from {csv_path}")


# ============================================================
# Main
# ============================================================

def run_experiments():
    openai_key = OPENAI_API_KEY.strip()

    with open(RUN_LOG_TXT, "w") as runlog:
        log_line(runlog, f"DATASETS: {DATASETS}")
        log_line(runlog, f"SIZES: {SIZES}")
        log_line(runlog, f"METHODS: {METHODS}")
        log_line(runlog, "-" * 90)

        if not openai_key and ("GPT35" in METHODS or "GPT4oMini" in METHODS):
            log_line(runlog, "[WARN] OPENAI_API_KEY is empty. GPT reranking will be skipped.")

        all_rows = []

        for dataset in DATASETS:
            dataset_start = time.time()
            log_line(runlog, "\n" + "=" * 90)
            log_line(runlog, f"[DATASET] {dataset}")
            log_line(runlog, "=" * 90)

            # NEW: collect per-query NDCG vectors per method per size for significance
            per_query_ndcg_by_method = {m: {} for m in METHODS}   # {method: {k: {qid: ndcg}}}

            for k in SIZES:
                size_start = time.time()
                log_line(runlog, f"\n[SIZE] Top-{k}")

                bm25_res, qrels, topics, searcher = retrieve_topk(dataset, k=k)

                queries = {}
                if isinstance(topics, dict):
                    for qid, q in topics.items():
                        if isinstance(q, dict):
                            qtext = q.get("title") or q.get("text") or str(list(q.values())[0])
                        else:
                            qtext = str(q)
                        queries[qid] = qtext
                        queries[str(qid)] = qtext
                else:
                    for i, q in enumerate(topics):
                        qtext = str(q)
                        queries[i] = qtext
                        queries[str(i)] = qtext

                def doc_fetcher(docid: str) -> str:
                    return get_doc_text(searcher, str(docid))

                # -----------------------------
                # BM25
                # -----------------------------
                if "BM25" in METHODS:
                    bm25_run = rerank_bm25(bm25_res, k=k)
                    bm25_ndcg = compute_mean_ndcg10(bm25_run, qrels)

                    in_docids, out_docids = _build_per_query_io_docids_from_run(bm25_res, bm25_run, k)
                    bm25_pq = compute_per_query_ndcg10_from_docids(out_docids, qrels)
                    per_query_ndcg_by_method["BM25"][k] = bm25_pq

                    # per-qid logs (input/output/ndcg)
                    for qid_str, nd in bm25_pq.items():
                        qtext = queries.get(qid_str, queries.get(int(qid_str), "")) if qid_str.isdigit() else queries.get(qid_str, "")
                        save_qid_log_with_ndcg(
                            root=LOGS_ROOT, k=k, dataset=dataset, method="BM25",
                            qid=qid_str, query=str(qtext),
                            input_docids=in_docids.get(qid_str, []),
                            output_docids=out_docids.get(qid_str, []),
                            ndcg10=float(nd)
                        )

                    # per-size per-query ndcg summary
                    save_per_query_result_log(RESULTS_ROOT, dataset, "BM25", k, bm25_pq, bm25_ndcg)

                    log_line(runlog, f"[RESULT] {dataset} | k={k} | BM25 | nDCG@10 = {bm25_ndcg:.4f}")
                    all_rows.append((dataset, k, "BM25", bm25_ndcg))

                # -----------------------------
                # MonoBERT
                # -----------------------------
                if "MonoBERT" in METHODS:
                    monoBERT_run = rerank_supervised_models("MonoBERT", bm25_res, queries, doc_fetcher, batch_size=8)
                    monoBERT_ndcg = compute_mean_ndcg10(monoBERT_run, qrels)

                    in_docids, out_docids = _build_per_query_io_docids_from_run(bm25_res, monoBERT_run, k)
                    monoBERT_pq = compute_per_query_ndcg10_from_docids(out_docids, qrels)
                    per_query_ndcg_by_method["MonoBERT"][k] = monoBERT_pq

                    for qid_str, nd in monoBERT_pq.items():
                        qtext = queries.get(qid_str, queries.get(int(qid_str), "")) if qid_str.isdigit() else queries.get(qid_str, "")
                        save_qid_log_with_ndcg(
                            root=LOGS_ROOT, k=k, dataset=dataset, method="MonoBERT",
                            qid=qid_str, query=str(qtext),
                            input_docids=in_docids.get(qid_str, []),
                            output_docids=out_docids.get(qid_str, []),
                            ndcg10=float(nd)
                        )

                    save_per_query_result_log(RESULTS_ROOT, dataset, "MonoBERT", k, monoBERT_pq, monoBERT_ndcg)

                    log_line(runlog, f"[RESULT] {dataset} | k={k} | MonoBERT | nDCG@10 = {monoBERT_ndcg:.4f}")
                    all_rows.append((dataset, k, "MonoBERT", monoBERT_ndcg))

                # -----------------------------
                # MonoT5 220M
                # -----------------------------
                if "MonoT5_220M" in METHODS:
                    t5base_run = rerank_supervised_models("MonoT5_220M", bm25_res, queries, doc_fetcher, batch_size=8)
                    t5base_ndcg = compute_mean_ndcg10(t5base_run, qrels)

                    in_docids, out_docids = _build_per_query_io_docids_from_run(bm25_res, t5base_run, k)
                    t5base_pq = compute_per_query_ndcg10_from_docids(out_docids, qrels)
                    per_query_ndcg_by_method["MonoT5_220M"][k] = t5base_pq

                    for qid_str, nd in t5base_pq.items():
                        qtext = queries.get(qid_str, queries.get(int(qid_str), "")) if qid_str.isdigit() else queries.get(qid_str, "")
                        save_qid_log_with_ndcg(
                            root=LOGS_ROOT, k=k, dataset=dataset, method="MonoT5_220M",
                            qid=qid_str, query=str(qtext),
                            input_docids=in_docids.get(qid_str, []),
                            output_docids=out_docids.get(qid_str, []),
                            ndcg10=float(nd)
                        )

                    save_per_query_result_log(RESULTS_ROOT, dataset, "MonoT5_220M", k, t5base_pq, t5base_ndcg)

                    log_line(runlog, f"[RESULT] {dataset} | k={k} | MonoT5_220M | nDCG@10 = {t5base_ndcg:.4f}")
                    all_rows.append((dataset, k, "MonoT5_220M", t5base_ndcg))

                # -----------------------------
                # MonoT5 3B
                # -----------------------------
                if "MonoT5_3B" in METHODS:
                    t53b_run = rerank_supervised_models("MonoT5_3B", bm25_res, queries, doc_fetcher, batch_size=4)
                    t53b_ndcg = compute_mean_ndcg10(t53b_run, qrels)

                    in_docids, out_docids = _build_per_query_io_docids_from_run(bm25_res, t53b_run, k)
                    t53b_pq = compute_per_query_ndcg10_from_docids(out_docids, qrels)
                    per_query_ndcg_by_method["MonoT5_3B"][k] = t53b_pq

                    for qid_str, nd in t53b_pq.items():
                        qtext = queries.get(qid_str, queries.get(int(qid_str), "")) if qid_str.isdigit() else queries.get(qid_str, "")
                        save_qid_log_with_ndcg(
                            root=LOGS_ROOT, k=k, dataset=dataset, method="MonoT5_3B",
                            qid=qid_str, query=str(qtext),
                            input_docids=in_docids.get(qid_str, []),
                            output_docids=out_docids.get(qid_str, []),
                            ndcg10=float(nd)
                        )

                    save_per_query_result_log(RESULTS_ROOT, dataset, "MonoT5_3B", k, t53b_pq, t53b_ndcg)

                    log_line(runlog, f"[RESULT] {dataset} | k={k} | MonoT5_3B | nDCG@10 = {t53b_ndcg:.4f}")
                    all_rows.append((dataset, k, "MonoT5_3B", t53b_ndcg))

                # -----------------------------
                # mMARCO-CE
                # -----------------------------
                if "mMARCO_CE" in METHODS:
                    ce_run = rerank_mmarco_ce(bm25_res, queries, doc_fetcher, batch_size=16)
                    ce_ndcg = compute_mean_ndcg10(ce_run, qrels)

                    in_docids, out_docids = _build_per_query_io_docids_from_run(bm25_res, ce_run, k)
                    ce_pq = compute_per_query_ndcg10_from_docids(out_docids, qrels)
                    per_query_ndcg_by_method["mMARCO_CE"][k] = ce_pq

                    for qid_str, nd in ce_pq.items():
                        qtext = queries.get(qid_str, queries.get(int(qid_str), "")) if qid_str.isdigit() else queries.get(qid_str, "")
                        save_qid_log_with_ndcg(
                            root=LOGS_ROOT, k=k, dataset=dataset, method="mMARCO_CE",
                            qid=qid_str, query=str(qtext),
                            input_docids=in_docids.get(qid_str, []),
                            output_docids=out_docids.get(qid_str, []),
                            ndcg10=float(nd)
                        )

                    save_per_query_result_log(RESULTS_ROOT, dataset, "mMARCO_CE", k, ce_pq, ce_ndcg)

                    log_line(runlog, f"[RESULT] {dataset} | k={k} | mMARCO-CE | nDCG@10 = {ce_ndcg:.4f}")
                    all_rows.append((dataset, k, "mMARCO_CE", ce_ndcg))

                # -----------------------------
                # GPT rerankers
                # -----------------------------
                if openai_key:
                    if "GPT35" in METHODS:
                        gpt35_run, gpt35_out_docids = rerank_rankgpt_llm(
                            dataset=dataset,
                            bm25_res=bm25_res,
                            topics=topics,
                            searcher=searcher,
                            k=k,
                            model_name="gpt-3.5-turbo",
                            openai_key=openai_key
                        )
                        gpt35_ndcg = compute_mean_ndcg10(gpt35_run, qrels)

                        # input/output/ndcg logs per qid
                        gpt35_pq = compute_per_query_ndcg10_from_docids(gpt35_out_docids, qrels)
                        per_query_ndcg_by_method["GPT35"][k] = gpt35_pq

                        for qid_str, nd in gpt35_pq.items():
                            # input = BM25 top-k
                            items = bm25_res.get(qid_str, None)
                            if items is None:
                                try:
                                    items = bm25_res.get(int(qid_str), [])
                                except Exception:
                                    items = []
                            input_docids = [str(d) for (d, _) in (items[:k] if items else [])]
                            qtext = queries.get(qid_str, queries.get(int(qid_str), "")) if qid_str.isdigit() else queries.get(qid_str, "")
                            save_qid_log_with_ndcg(
                                root=LOGS_ROOT, k=k, dataset=dataset, method="gpt-3.5-turbo",
                                qid=qid_str, query=str(qtext),
                                input_docids=input_docids,
                                output_docids=[str(d) for d in gpt35_out_docids.get(qid_str, [])],
                                ndcg10=float(nd)
                            )

                        save_per_query_result_log(RESULTS_ROOT, dataset, "GPT35", k, gpt35_pq, gpt35_ndcg)

                        log_line(runlog, f"[RESULT] {dataset} | k={k} | GPT-3.5-turbo | nDCG@10 = {gpt35_ndcg:.4f}")
                        all_rows.append((dataset, k, "GPT35", gpt35_ndcg))

                    if "GPT4oMini" in METHODS:
                        gpt4mini_run, gpt4mini_out_docids = rerank_rankgpt_llm(
                            dataset=dataset,
                            bm25_res=bm25_res,
                            topics=topics,
                            searcher=searcher,
                            k=k,
                            model_name="gpt-4o-mini",
                            openai_key=openai_key
                        )
                        gpt4mini_ndcg = compute_mean_ndcg10(gpt4mini_run, qrels)

                        gpt4mini_pq = compute_per_query_ndcg10_from_docids(gpt4mini_out_docids, qrels)
                        per_query_ndcg_by_method["GPT4oMini"][k] = gpt4mini_pq

                        for qid_str, nd in gpt4mini_pq.items():
                            items = bm25_res.get(qid_str, None)
                            if items is None:
                                try:
                                    items = bm25_res.get(int(qid_str), [])
                                except Exception:
                                    items = []
                            input_docids = [str(d) for (d, _) in (items[:k] if items else [])]
                            qtext = queries.get(qid_str, queries.get(int(qid_str), "")) if qid_str.isdigit() else queries.get(qid_str, "")
                            save_qid_log_with_ndcg(
                                root=LOGS_ROOT, k=k, dataset=dataset, method="gpt-4o-mini",
                                qid=qid_str, query=str(qtext),
                                input_docids=input_docids,
                                output_docids=[str(d) for d in gpt4mini_out_docids.get(qid_str, [])],
                                ndcg10=float(nd)
                            )

                        save_per_query_result_log(RESULTS_ROOT, dataset, "GPT4oMini", k, gpt4mini_pq, gpt4mini_ndcg)

                        log_line(runlog, f"[RESULT] {dataset} | k={k} | GPT-4o-mini | nDCG@10 = {gpt4mini_ndcg:.4f}")
                        all_rows.append((dataset, k, "GPT4oMini", gpt4mini_ndcg))

                size_time = time.time() - size_start
                log_line(runlog, f"[TIME] {dataset} | Top-{k} finished in {size_time/60:.2f} min")

            dataset_time = time.time() - dataset_start
            log_line(runlog, f"[TOTAL TIME] Dataset {dataset} finished in {dataset_time/60:.2f} min")

            # ====================================================
            # ✅ NEW: Save results per-dataset immediately
            # ====================================================
            df_dataset = pd.DataFrame(
                [r for r in all_rows if r[0] == dataset],
                columns=["dataset", "k", "method", "ndcg@10"]
            )

            OUT_CSV_DS   = os.path.join(RESULTS_ROOT, f"ndcg_trend_all_{dataset}.csv")
            PIVOT_CSV_DS = os.path.join(RESULTS_ROOT, f"ndcg_trend_pivot_{dataset}.csv")
            SUMMARY_DS   = os.path.join(RESULTS_ROOT, f"summary_{dataset}.txt")

            df_dataset.to_csv(OUT_CSV_DS, index=False)
            log_line(runlog, f"[SAVED] Results CSV -> {OUT_CSV_DS}")

            pivot = df_dataset.pivot_table(index=["dataset", "method"], columns="k", values="ndcg@10", aggfunc="mean")
            pivot.to_csv(PIVOT_CSV_DS)
            log_line(runlog, f"[SAVED] Pivot CSV -> {PIVOT_CSV_DS}")

            with open(SUMMARY_DS, "w") as sf:
                sf.write(f"DATASETS: [{dataset}]\n")
                sf.write(f"SIZES: {SIZES}\n")
                sf.write(f"METHODS: {METHODS}\n\n")
                sf.write("=== nDCG@10 Pivot ===\n\n")
                sf.write(pivot.to_string())
                sf.write("\n")

            log_line(runlog, f"[SAVED] Summary TXT -> {SUMMARY_DS}")

            # plot only this dataset
            plot_trends(df_dataset)
            log_line(runlog, f"[DONE] Plot saved into: {PLOTS_ROOT}")

            # ====================================================
            # ✅ NEW: Significance per (dataset, model) across sizes
            # ====================================================
            # saved after finishing each dataset (for all its models)
            for method in METHODS:
                if method not in per_query_ndcg_by_method:
                    continue

                got_sizes = sorted(per_query_ndcg_by_method[method].keys())
                if len(got_sizes) < 2:
                    continue

                sizes_to_test = [kk for kk in SIZES if kk in per_query_ndcg_by_method[method]]
                if len(sizes_to_test) < 2:
                    sizes_to_test = got_sizes

                sig_obj = {
                    "dataset": dataset,
                    "method": method,
                    "sizes_tested": sizes_to_test,
                    "scipy_available": _HAVE_SCIPY,
                    "note": "Paired Wilcoxon for each size-pair + overall Friedman across sizes (requires scipy).",
                    "results": run_significance_by_size(per_query_ndcg_by_method[method], sizes_to_test),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                save_significance_result(RESULTS_ROOT, dataset, method, sig_obj)
                log_line(runlog, f"[SAVED] Significance -> {RESULTS_ROOT}/significance_by_size/{dataset}/{method}_significance.json")



if __name__ == "__main__":
    # 1) Run the full pipeline ONCE:
    run_experiments()

    # 2) Later, ONLY re-run plotting anytime :
#     datasets_to_plot = ["nfcorpus"]
#     plot_datasets_from_csv(datasets_to_plot)
