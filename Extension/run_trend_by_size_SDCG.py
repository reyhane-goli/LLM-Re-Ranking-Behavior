OPENAI_API_KEY = ""

# ============================================================
# run_trend_by_size_SDCG.py  (UPDATED: qrels/run normalization + bundle logging)
# ============================================================

import os
import json
import time
import math
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from matplotlib.patches import Patch


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
from evaluate import eval_ndcg10  # kept (we won't use it for the new NDCG definition)

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
#     "dl19",
#     "trec-covid",
#     "webis-touche2020",
#     "scifact",
    "trec-news",
]

# You can choose: [10, 20, 50, 200] (or any list)
SIZES = [10, 20, 50, 100, 200]
# SIZES = [10, 20]

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
#     "MonoT5_3B",
    "GPT35",
    "GPT4oMini",
]


# ============================================================
# OUTPUT ROOTS
# ============================================================

RESULTS_ROOT = "./results_by_size"
LOGS_ROOT    = "./logs_llm_rerank_sizes"
PLOTS_ROOT   = "./plots_by_size"

# NEW: cache so you can skip reranking next time
CACHE_ROOT   = "./cache_by_size"

os.makedirs(RESULTS_ROOT, exist_ok=True)
os.makedirs(LOGS_ROOT, exist_ok=True)
os.makedirs(PLOTS_ROOT, exist_ok=True)
os.makedirs(CACHE_ROOT, exist_ok=True)

# NEW: make parallel terminals safe (unique run tag per process)
RUN_TAG = os.environ.get("RUN_TAG", time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}")
RUN_LOG_TXT  = os.path.join(RESULTS_ROOT, f"run_log_{RUN_TAG}.txt")


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def log_line(fp, msg: str):
    fp.write(msg.rstrip() + "\n")
    fp.flush()

def _safe_write_json(path: str, obj: Any):
    """Atomic-ish write to avoid corruption if two terminals finish close together."""
    ensure_dir(os.path.dirname(path))
    tmp = path + f".tmp_{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def _safe_read_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

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
                 input_scores=None, output_scores=None,
                 extra: Optional[Dict[str, Any]] = None):
    out_dir = os.path.join(root, dataset, f"k{k}", method)
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
    if extra:
        obj.update(extra)

    _safe_write_json(out_path, obj)

def make_scores_descending(order_docids: List[str]) -> Dict[str, float]:
    scores = {}
    n = len(order_docids)
    for i, did in enumerate(order_docids):
        scores[str(did)] = float(n - i)
    return scores


# ============================================================
# NEW: Normalization (FIX for DL19 NDCG=0 due to qid/docid type mismatch)
# ============================================================

def normalize_qrels(qrels_in: Dict[Any, Dict[Any, Any]]) -> Dict[str, Dict[str, int]]:
    """Force qrels into Dict[str, Dict[str, int]] so qid/docid matching works."""
    out: Dict[str, Dict[str, int]] = {}
    if not qrels_in:
        return out
    for qid, docrels in qrels_in.items():
        qid_s = str(qid)
        out[qid_s] = {}
        if not docrels:
            continue
        for did, r in docrels.items():
            try:
                out[qid_s][str(did)] = int(r)
            except Exception:
                out[qid_s][str(did)] = 0
    return out

def normalize_run_docids(run_docids_in: Dict[Any, List[Any]]) -> Dict[str, List[str]]:
    """Force run into Dict[str, List[str]]"""
    if not run_docids_in:
        return {}
    return {str(qid): [str(d) for d in (docids or [])] for qid, docids in run_docids_in.items()}

def debug_alignment(dataset: str, k: int, method: str,
                    run_docids: Dict[str, List[str]],
                    qrels: Dict[str, Dict[str, int]],
                    runlog_fp,
                    n_examples: int = 2):
    run_qids = set(run_docids.keys())
    qrels_qids = set(qrels.keys())
    overlap = sorted(list(run_qids & qrels_qids))

    log_line(runlog_fp, f"[DEBUG] {dataset} k={k} {method}: run_qids={len(run_qids)} qrels_qids={len(qrels_qids)} overlap={len(overlap)}")

    if not overlap:
        log_line(runlog_fp, f"[DEBUG] {dataset} k={k} {method}: OVERLAP=0 -> NDCG will be 0.0 (qid mismatch).")
        return

    for qid in overlap[:n_examples]:
        run_docs = run_docids.get(qid, [])[:5]
        qrels_docs = list(qrels[qid].keys())[:5]
        log_line(runlog_fp, f"[DEBUG] example qid={qid}: run_docids[:5]={run_docs}")
        log_line(runlog_fp, f"[DEBUG] example qid={qid}: qrels_docids[:5]={qrels_docs}")


# ============================================================
# NDCG@10 definition (constant IDCG, gain = 2^rel-1)
# ============================================================

TOP_K_EVAL = 10

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

def idcg_constant(max_grade: int, k: int = TOP_K_EVAL) -> float:
    # trec_eval / pytrec_eval gain = (2^rel - 1)
    # constant ideal = k docs at max_grade
    if k <= 0:
        return 0.0
    gain = float((2 ** int(max_grade)) - 1)
    s = 0.0
    for i in range(k):
        s += gain / math.log2(i + 2)
    return s

def dcg_2pow(rels: List[int], k: int = TOP_K_EVAL) -> float:
    # trec_eval / pytrec_eval gain = (2^rel - 1)
    s = 0.0
    for i, r in enumerate(rels[:k]):
        s += ((2 ** int(r)) - 1) / math.log2(i + 2)
    return s

def ndcg_const_for_qid(docids_in_rank_order: List[str],
                       qrels_for_qid: Dict[str, int],
                       idcg_const: float,
                       k: int = TOP_K_EVAL) -> float:
    if idcg_const <= 0:
        return 0.0
    rels = []
    for did in docids_in_rank_order[:k]:
        rels.append(int(qrels_for_qid.get(str(did), 0)))
    while len(rels) < k:
        rels.append(0)
    return float(dcg_2pow(rels, k=k) / idcg_const)

def build_per_query_ndcg(run_docids: Dict[str, List[str]],
                         qrels: Dict[str, Dict[str, int]],
                         idcg_const: float,
                         k: int = TOP_K_EVAL) -> Dict[str, float]:
    out = {}
    # evaluate on qids that exist in run_docids AND qrels
    common = [qid for qid in run_docids.keys() if str(qid) in qrels]
    for qid in common:
        out[str(qid)] = ndcg_const_for_qid(run_docids[str(qid)], qrels[str(qid)], idcg_const, k=k)
    return out

def mean_of_dict(vals: Dict[str, float]) -> float:
    if not vals:
        return 0.0
    return float(sum(vals.values()) / len(vals))


# ============================================================
# Cache helpers (so reranking doesn't re-run)
# ============================================================

def cache_dir(dataset: str, k: int, method: str) -> str:
    return os.path.join(CACHE_ROOT, dataset, f"k{k}", method)

def cache_run_path(dataset: str, k: int, method: str) -> str:
    return os.path.join(cache_dir(dataset, k, method), "run.json")

def cache_inputs_path(dataset: str, k: int) -> str:
    return os.path.join(CACHE_ROOT, dataset, f"k{k}", "BM25", "inputs.json")

def cache_maxgrade_path(dataset: str) -> str:
    return os.path.join(CACHE_ROOT, dataset, "max_grade.json")

def cache_idcg_path(dataset: str) -> str:
    return os.path.join(CACHE_ROOT, dataset, "idcg_const.json")

def load_cached_run_docids(dataset: str, k: int, method: str) -> Optional[Dict[str, List[str]]]:
    p = cache_run_path(dataset, k, method)
    if not os.path.exists(p):
        return None
    obj = _safe_read_json(p)
    return normalize_run_docids(obj)

def save_cached_run_docids(dataset: str, k: int, method: str, run_docids: Dict[str, List[str]]):
    p = cache_run_path(dataset, k, method)
    _safe_write_json(p, normalize_run_docids(run_docids))


# ============================================================
# NEW: One-file bundle per (dataset, k, method) with input+output
# ============================================================

def bundle_path(dataset: str, k: int, method: str) -> str:
    return os.path.join(cache_dir(dataset, k, method), "bundle.json")

def save_bundle(dataset: str, k: int, method: str,
                queries: Dict[Any, str],
                input_docids: Dict[str, List[str]],
                output_docids: Dict[str, List[str]],
                extra: Optional[Dict[str, Any]] = None):
    input_docids = normalize_run_docids(input_docids)
    output_docids = normalize_run_docids(output_docids)

    qout = {}
    for qid in output_docids.keys():
        qtext = ""
        if qid in queries:
            qtext = str(queries[qid])
        else:
            try:
                qtext = str(queries[int(qid)])
            except Exception:
                qtext = ""

        qout[qid] = {
            "query": qtext,
            "input_docids": input_docids.get(qid, []),
            "output_docids": output_docids.get(qid, []),
        }

    obj = {
        "dataset": dataset,
        "k": int(k),
        "method": method,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "qids": qout,
    }
    if extra:
        obj.update(extra)

    _safe_write_json(bundle_path(dataset, k, method), obj)

def load_bundle(dataset: str, k: int, method: str) -> Optional[Dict[str, Any]]:
    p = bundle_path(dataset, k, method)
    if not os.path.exists(p):
        return None
    return _safe_read_json(p)


# ============================================================
# Rerankers (unchanged behavior, but now cache input/output)
# ============================================================

def rerank_bm25(bm25_res: Dict[str, List[Tuple[str, float]]], k: int) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[str]]]:
    run = {}
    run_docids = {}
    for qid, items in bm25_res.items():
        run[str(qid)] = {str(docid): float(score) for (docid, score) in items[:k]}
        run_docids[str(qid)] = [str(docid) for (docid, _) in items[:k]]
    return run, run_docids

def rerank_supervised_models(method: str, bm25_res, queries, doc_fetcher, batch_size=8) -> Dict[str, List[Tuple[str, float]]]:
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

    return rr  # dict[qid] -> list[(docid, score)]

def rerank_mmarco_ce(bm25_res, queries, doc_fetcher, batch_size=16) -> Dict[str, List[Tuple[str, float]]]:
    scorer = CrossEncoderReRanker(MMARCO_CE_CKPT)
    rr = rerank_unsup("mMARCO-CE", scorer, bm25_res, queries, doc_fetcher,
                      batch_size=batch_size, colour="yellow")
    return rr

def rerank_rankgpt_llm(dataset: str, bm25_res, topics, searcher, k: int,
                       model_name: str, openai_key: str) -> Dict[str, List[str]]:
    """
    Returns run_docids: {qid: [docid1, docid2, ...]} in final rank order.
    Uses caching per-(dataset,k,model_name) so it won't re-call API if cached run exists.
    """
    cached = load_cached_run_docids(dataset, k, model_name)
    if cached is not None:
        return cached

    run_docids: Dict[str, List[str]] = {}

    def safe_get_items(qid_str: str):
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
        out_docids = [str(h["docid"]) for h in reranked_hits]
        run_docids[qid_str] = out_docids

        save_qid_log(
            root=LOGS_ROOT,
            k=k,
            dataset=dataset,
            method=model_name,
            qid=qid_str,
            query=query,
            input_docids=[str(d) for (d, _) in items],
            output_docids=out_docids,
        )

    save_cached_run_docids(dataset, k, model_name, run_docids)
    return run_docids


# ============================================================
# Plotting (unchanged)
# ============================================================

def plot_dataset_line_and_box(
    dataset: str,
    methods: List[str],
    sizes: List[int],
    per_query_store: Dict[str, Dict[int, Dict[str, float]]],
    out_pdf: str,
    use_log_x: bool = True,
):
    LINE_COLORS = {
        "BM25":        "#BDBDBD",
        "MonoBERT":    "#4DB6AC",
        "MonoT5_220M": "#FFB74D",
        "MonoT5_3B":   "#64B5F6",
        "mMARCO_CE":   "#81C784",
        "GPT35":       "#E57373",
        "GPT4oMini":   "#BA68C8",
    }
    DOT_COLORS = {
        "BM25":        "#424242",
        "MonoBERT":    "#00695C",
        "MonoT5_220M": "#E65100",
        "MonoT5_3B":   "#0D47A1",
        "mMARCO_CE":   "#1B5E20",
        "GPT35":       "#B71C1C",
        "GPT4oMini":   "#4A148C",
    }

    DISPLAY_NAME = {
        "GPT35": "GPT-3.5-turbo",
        "GPT4oMini": "GPT-4o-mini",
        "MonoT5_3B": "MonoT5-3B",
        "MonoT5_220M": "MonoT5-220M",
        "mMARCO_CE": "mMARCO-CE",
        "MonoBERT": "MonoBERT",
        "BM25": "BM25",
    }

    def pretty(m: str) -> str:
        return DISPLAY_NAME.get(m, m.replace("_", "-"))

    LEGEND_ORDER = ["GPT4oMini", "MonoT5_3B", "MonoT5_220M", "mMARCO_CE", "GPT35", "MonoBERT", "BM25"]
    present = [m for m in LEGEND_ORDER if m in methods]
    present += [m for m in methods if m not in present]
    methods = present

    xs = []
    for k in sizes:
        xs.append(math.log10(k) if use_log_x else float(k))

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    n_methods = max(1, len(methods))
    width = 0.07 if use_log_x else 1.0
    offsets = [ (i - (n_methods - 1)/2) * width for i in range(n_methods) ]

    for mi, method in enumerate(methods):
        method_store = per_query_store.get(method, {})
        data = []
        positions = []
        for si, k in enumerate(sizes):
            qmap = method_store.get(int(k), {})
            vals = [float(v) for v in qmap.values()]
            if not vals:
                continue
            data.append(vals)
            positions.append(xs[si] + offsets[mi])

        if data:
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=(width * 0.9 if use_log_x else 0.7),
                patch_artist=True,
                manage_ticks=False,
                showfliers=False
            )
            for box in bp["boxes"]:
                box.set_facecolor(LINE_COLORS.get(method, "#BDBDBD"))
                box.set_edgecolor(DOT_COLORS.get(method, "#424242"))
                box.set_linewidth(1.2)
            for whisker in bp["whiskers"]:
                whisker.set_color(DOT_COLORS.get(method, "#424242"))
                whisker.set_linewidth(1.0)
            for cap in bp["caps"]:
                cap.set_color(DOT_COLORS.get(method, "#424242"))
                cap.set_linewidth(1.0)
            for median in bp["medians"]:
                median.set_color(DOT_COLORS.get(method, "#424242"))
                median.set_linewidth(1.4)

    for method in methods:
        ys = []
        x_use = []
        for si, k in enumerate(sizes):
            qmap = per_query_store.get(method, {}).get(int(k), {})
            if not qmap:
                continue
            ys.append(sum(qmap.values()) / len(qmap))
            x_use.append(xs[si])

        if ys:
            ax.plot(
                x_use,
                ys,
                marker="o",
                markersize=6.5,
                linewidth=2.2,
                color=LINE_COLORS.get(method, "#BDBDBD"),
                markerfacecolor=DOT_COLORS.get(method, "#424242"),
                markeredgecolor=DOT_COLORS.get(method, "#424242"),
                label=pretty(method)
            )

    ax.set_ylabel("NDCG@10", fontsize=12)
    ax.set_xlabel("Input Size", fontsize=12)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in sizes], fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True, fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.80, 1])

    ensure_dir(os.path.dirname(out_pdf))
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print("[PLOT SAVED]", out_pdf)


# ============================================================
# Main
# ============================================================

def run_experiments():
    openai_key = OPENAI_API_KEY.strip()

    with open(RUN_LOG_TXT, "w") as runlog:
        log_line(runlog, f"RUN_TAG: {RUN_TAG}")
        log_line(runlog, f"DATASETS: {DATASETS}")
        log_line(runlog, f"SIZES: {SIZES}")
        log_line(runlog, f"METHODS: {METHODS}")
        log_line(runlog, "-" * 90)

        if not openai_key and ("GPT35" in METHODS or "GPT4oMini" in METHODS):
            log_line(runlog, "[WARN] OPENAI_API_KEY is empty. GPT reranking will be skipped.")

        for dataset in DATASETS:
            dataset_start = time.time()
            log_line(runlog, "\n" + "=" * 90)
            log_line(runlog, f"[DATASET] {dataset}")
            log_line(runlog, "=" * 90)

            ds_out_dir = os.path.join(RESULTS_ROOT, dataset, RUN_TAG)
            ensure_dir(ds_out_dir)

            ds_log_dir = os.path.join(LOGS_ROOT, dataset, RUN_TAG)
            ensure_dir(ds_log_dir)

            ds_plot_dir = os.path.join(PLOTS_ROOT, dataset)
            ensure_dir(ds_plot_dir)

            details_rows = []
            summary_rows = []
            per_query_store: Dict[str, Dict[int, Dict[str, float]]] = {}

            maxg = None
            idcg_c = None

            for k in SIZES:
                size_start = time.time()
                log_line(runlog, f"\n[SIZE] Top-{k}")

                bm25_res, qrels, topics, searcher = retrieve_topk(dataset, k=k)

                # ✅ FIX: normalize qrels so qid matching works (prevents common=[] -> mean=0.0)
                qrels = normalize_qrels(qrels)

                # max_grade + idcg_const (per dataset)
                if maxg is None or idcg_c is None:
                    mg_path = cache_maxgrade_path(dataset)
                    id_path = cache_idcg_path(dataset)
                    if os.path.exists(mg_path) and os.path.exists(id_path):
                        try:
                            maxg = int(_safe_read_json(mg_path)["max_grade"])
                            idcg_c = float(_safe_read_json(id_path)["idcg_const@10"])
                        except Exception:
                            maxg = None
                            idcg_c = None

                    if maxg is None or idcg_c is None:
                        maxg = max_grade_from_qrels(qrels)
                        idcg_c = idcg_constant(maxg, k=TOP_K_EVAL)
                        _safe_write_json(mg_path, {"dataset": dataset, "max_grade": int(maxg), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
                        _safe_write_json(id_path, {"dataset": dataset, "idcg_const@10": float(idcg_c), "max_grade": int(maxg), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})

                # Build queries dict (kept)
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

                # BM25 input candidates (top-k) for bundle/logging
                bm25_inputs = {str(qid): [str(d) for (d, _) in items[:k]] for qid, items in bm25_res.items()}
                bm25_inputs = normalize_run_docids(bm25_inputs)

                # --------------------------------------------------
                # BM25
                # --------------------------------------------------
                if "BM25" in METHODS:
                    bm25_run, bm25_docids = rerank_bm25(bm25_res, k=k)
                    bm25_docids = normalize_run_docids(bm25_docids)

                    inp_path = cache_inputs_path(dataset, k)
                    if not os.path.exists(inp_path):
                        _safe_write_json(inp_path, bm25_docids)

                    save_cached_run_docids(dataset, k, "BM25", bm25_docids)

#                     debug_alignment(dataset, k, "BM25", bm25_docids, qrels, runlog)

                    per_q = build_per_query_ndcg(bm25_docids, qrels, idcg_c, k=TOP_K_EVAL)
                    per_query_store.setdefault("BM25", {})[int(k)] = per_q
                    mean_ndcg = mean_of_dict(per_q)

                    log_line(runlog, f"[RESULT] {dataset} | k={k} | BM25 | NDCG@10 = {mean_ndcg:.6f}")
                    summary_rows.append((dataset, k, "BM25", mean_ndcg))

                    # ✅ one-file bundle (dataset,k,method)
                    save_bundle(dataset, k, "BM25", queries, bm25_inputs, bm25_docids)

                    for qid, nd in per_q.items():
                        qtext = queries.get(qid, queries.get(int(qid), "")) if str(qid).isdigit() else queries.get(qid, "")
                        inp = bm25_docids.get(str(qid), [])
                        out = bm25_docids.get(str(qid), [])
                        details_rows.append((dataset, k, "BM25", str(qid), nd, json.dumps(inp), json.dumps(out), qtext))

                        save_qid_log(
                            root=ds_log_dir,
                            k=k,
                            dataset=dataset,
                            method="BM25",
                            qid=str(qid),
                            query=str(qtext),
                            input_docids=inp,
                            output_docids=out,
                        )

                # --------------------------------------------------
                # Supervised rerankers
                # --------------------------------------------------
                for sup_method in ["MonoBERT", "MonoT5_220M", "MonoT5_3B"]:
                    if sup_method not in METHODS:
                        continue

                    cached_docids = load_cached_run_docids(dataset, k, sup_method)
                    if cached_docids is None:
                        rr = rerank_supervised_models(
                            sup_method, bm25_res, queries, doc_fetcher,
                            batch_size=8 if sup_method != "MonoT5_3B" else 4
                        )
                        run_docids = {str(qid): [str(d) for (d, _) in pairs] for qid, pairs in rr.items()}
                        run_docids = normalize_run_docids(run_docids)
                        save_cached_run_docids(dataset, k, sup_method, run_docids)
                    else:
                        run_docids = normalize_run_docids(cached_docids)

#                     debug_alignment(dataset, k, sup_method, run_docids, qrels, runlog)

                    per_q = build_per_query_ndcg(run_docids, qrels, idcg_c, k=TOP_K_EVAL)
                    per_query_store.setdefault(sup_method, {})[int(k)] = per_q
                    mean_ndcg = mean_of_dict(per_q)

                    log_line(runlog, f"[RESULT] {dataset} | k={k} | {sup_method} | NDCG@10 = {mean_ndcg:.6f}")
                    summary_rows.append((dataset, k, sup_method, mean_ndcg))

                    # ✅ one-file bundle (dataset,k,method)
                    save_bundle(dataset, k, sup_method, queries, bm25_inputs, run_docids)

                    for qid, nd in per_q.items():
                        qtext = queries.get(qid, queries.get(int(qid), "")) if str(qid).isdigit() else queries.get(qid, "")
                        inp = bm25_inputs.get(str(qid), [])
                        out = run_docids.get(str(qid), [])
                        details_rows.append((dataset, k, sup_method, str(qid), nd, json.dumps(inp), json.dumps(out), qtext))

                        save_qid_log(
                            root=ds_log_dir,
                            k=k,
                            dataset=dataset,
                            method=sup_method,
                            qid=str(qid),
                            query=str(qtext),
                            input_docids=inp,
                            output_docids=out,
                        )

                # --------------------------------------------------
                # mMARCO-CE
                # --------------------------------------------------
                if "mMARCO_CE" in METHODS:
                    cached_docids = load_cached_run_docids(dataset, k, "mMARCO_CE")
                    if cached_docids is None:
                        rr = rerank_mmarco_ce(bm25_res, queries, doc_fetcher, batch_size=16)
                        run_docids = {str(qid): [str(d) for (d, _) in pairs] for qid, pairs in rr.items()}
                        run_docids = normalize_run_docids(run_docids)
                        save_cached_run_docids(dataset, k, "mMARCO_CE", run_docids)
                    else:
                        run_docids = normalize_run_docids(cached_docids)

#                     debug_alignment(dataset, k, "mMARCO_CE", run_docids, qrels, runlog)

                    per_q = build_per_query_ndcg(run_docids, qrels, idcg_c, k=TOP_K_EVAL)
                    per_query_store.setdefault("mMARCO_CE", {})[int(k)] = per_q
                    mean_ndcg = mean_of_dict(per_q)

                    log_line(runlog, f"[RESULT] {dataset} | k={k} | mMARCO_CE | NDCG@10 = {mean_ndcg:.6f}")
                    summary_rows.append((dataset, k, "mMARCO_CE", mean_ndcg))

                    # ✅ one-file bundle
                    save_bundle(dataset, k, "mMARCO_CE", queries, bm25_inputs, run_docids)

                    for qid, nd in per_q.items():
                        qtext = queries.get(qid, queries.get(int(qid), "")) if str(qid).isdigit() else queries.get(qid, "")
                        inp = bm25_inputs.get(str(qid), [])
                        out = run_docids.get(str(qid), [])
                        details_rows.append((dataset, k, "mMARCO_CE", str(qid), nd, json.dumps(inp), json.dumps(out), qtext))

                        save_qid_log(
                            root=ds_log_dir,
                            k=k,
                            dataset=dataset,
                            method="mMARCO_CE",
                            qid=str(qid),
                            query=str(qtext),
                            input_docids=inp,
                            output_docids=out,
                        )

                # --------------------------------------------------
                # GPT rerankers
                # --------------------------------------------------
                if openai_key:
                    if "GPT35" in METHODS:
                        docids = rerank_rankgpt_llm(
                            dataset=dataset,
                            bm25_res=bm25_res,
                            topics=topics,
                            searcher=searcher,
                            k=k,
                            model_name="gpt-3.5-turbo",
                            openai_key=openai_key
                        )
                        docids = normalize_run_docids(docids)
                        save_cached_run_docids(dataset, k, "GPT35", docids)

#                         debug_alignment(dataset, k, "GPT35", docids, qrels, runlog)

                        per_q = build_per_query_ndcg(docids, qrels, idcg_c, k=TOP_K_EVAL)
                        per_query_store.setdefault("GPT35", {})[int(k)] = per_q
                        mean_ndcg = mean_of_dict(per_q)

                        log_line(runlog, f"[RESULT] {dataset} | k={k} | GPT-3.5-turbo | NDCG@10 = {mean_ndcg:.6f}")
                        summary_rows.append((dataset, k, "GPT35", mean_ndcg))

                        save_bundle(dataset, k, "GPT35", queries, bm25_inputs, docids,
                                    extra={"model_name": "gpt-3.5-turbo"})

                        for qid, nd in per_q.items():
                            qtext = queries.get(qid, queries.get(int(qid), "")) if str(qid).isdigit() else queries.get(qid, "")
                            inp = bm25_inputs.get(str(qid), [])
                            out = docids.get(str(qid), [])
                            details_rows.append((dataset, k, "GPT35", str(qid), nd, json.dumps(inp), json.dumps(out), qtext))

                    if "GPT4oMini" in METHODS:
                        docids = rerank_rankgpt_llm(
                            dataset=dataset,
                            bm25_res=bm25_res,
                            topics=topics,
                            searcher=searcher,
                            k=k,
                            model_name="gpt-4o-mini",
                            openai_key=openai_key
                        )
                        docids = normalize_run_docids(docids)
                        save_cached_run_docids(dataset, k, "GPT4oMini", docids)

#                         debug_alignment(dataset, k, "GPT4oMini", docids, qrels, runlog)

                        per_q = build_per_query_ndcg(docids, qrels, idcg_c, k=TOP_K_EVAL)
                        per_query_store.setdefault("GPT4oMini", {})[int(k)] = per_q
                        mean_ndcg = mean_of_dict(per_q)

                        log_line(runlog, f"[RESULT] {dataset} | k={k} | GPT-4o-mini | NDCG@10 = {mean_ndcg:.6f}")
                        summary_rows.append((dataset, k, "GPT4oMini", mean_ndcg))

                        save_bundle(dataset, k, "GPT4oMini", queries, bm25_inputs, docids,
                                    extra={"model_name": "gpt-4o-mini"})

                        for qid, nd in per_q.items():
                            qtext = queries.get(qid, queries.get(int(qid), "")) if str(qid).isdigit() else queries.get(qid, "")
                            inp = bm25_inputs.get(str(qid), [])
                            out = docids.get(str(qid), [])
                            details_rows.append((dataset, k, "GPT4oMini", str(qid), nd, json.dumps(inp), json.dumps(out), qtext))

                size_time = time.time() - size_start
                log_line(runlog, f"[TIME] {dataset} | Top-{k} finished in {size_time/60:.2f} min")

            # Save outputs
            df_summary = pd.DataFrame(summary_rows, columns=["dataset", "k", "method", "ndcg@10"])
            summary_csv = os.path.join(ds_out_dir, f"ndcg_trend_all_{dataset}.csv")
            df_summary.to_csv(summary_csv, index=False)
            log_line(runlog, f"[SAVED] Summary CSV -> {summary_csv}")

            pivot = df_summary.pivot_table(index=["dataset", "method"], columns="k", values="ndcg@10", aggfunc="mean")
            pivot_csv = os.path.join(ds_out_dir, f"ndcg_trend_pivot_{dataset}.csv")
            pivot.to_csv(pivot_csv)
            log_line(runlog, f"[SAVED] Pivot CSV -> {pivot_csv}")

            df_details = pd.DataFrame(
                details_rows,
                columns=["dataset", "k", "method", "qid", "ndcg@10", "input_docids_json", "output_docids_json", "query"]
            )
            details_csv = os.path.join(ds_out_dir, f"details_{dataset}.csv")
            df_details.to_csv(details_csv, index=False)
            log_line(runlog, f"[SAVED] Details CSV -> {details_csv}")

            plot_pdf = os.path.join(ds_plot_dir, f"{dataset}_line_and_box_{RUN_TAG}.pdf")
            plot_dataset_line_and_box(
                dataset=dataset,
                methods=[m for m in METHODS if m in per_query_store],
                sizes=SIZES,
                per_query_store=per_query_store,
                out_pdf=plot_pdf,
                use_log_x=True
            )
            log_line(runlog, f"[DONE] Plot saved -> {plot_pdf}")

            dataset_time = time.time() - dataset_start
            log_line(runlog, f"[TOTAL TIME] Dataset {dataset} finished in {dataset_time/60:.2f} min")


######################################## Plotting Part ################################################

def mean_ci95(vals, use_bootstrap: bool = True, n_boot: int = 2000, seed: int = 1234):
    vals = np.asarray(list(vals), dtype=float)
    if vals.size == 0:
        return (0.0, 0.0, 0.0)
    m = float(vals.mean())

    if not use_bootstrap:
        if vals.size == 1:
            return (m, m, m)
        se = float(vals.std(ddof=1) / np.sqrt(vals.size))
        lo = m - 1.96 * se
        hi = m + 1.96 * se
        return (m, lo, hi)

    rng = np.random.default_rng(seed)
    boots = []
    n = vals.size
    for _ in range(int(n_boot)):
        sample = rng.choice(vals, size=n, replace=True)
        boots.append(float(sample.mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (m, float(lo), float(hi))


def plot_mean_lines_with_ci_band(
    dataset: str,
    methods: list,
    sizes: list,
    per_query_store: dict,
    out_pdf: str,
    use_log_x: bool = True,
    bootstrap_ci: bool = True,
):
    xs = [math.log10(k) if use_log_x else float(k) for k in sizes]
    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    for method in methods:
        x_use, means, los, his = [], [], [], []
        for si, k in enumerate(sizes):
            qmap = per_query_store.get(method, {}).get(int(k), {})
            vals = list(qmap.values())
            if not vals:
                continue
            m, lo, hi = mean_ci95(vals, use_bootstrap=bootstrap_ci)
            x_use.append(xs[si])
            means.append(m)
            los.append(lo)
            his.append(hi)

        if not means:
            continue

        ax.plot(x_use, means, marker="o", linewidth=2.0, label=method)
        ax.fill_between(x_use, los, his, alpha=0.20)

    ax.set_ylabel("NDCG@10", fontsize=12)
    ax.set_xlabel("Candidate size (N)", fontsize=12)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in sizes], fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True, fontsize=10)

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    ensure_dir(os.path.dirname(out_pdf))
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print("[PLOT SAVED]", out_pdf)


def plot_mean_lines_with_point_ci(
    dataset: str,
    methods: list,
    sizes: list,
    per_query_store: dict,
    out_pdf: str,
    use_log_x: bool = True,
    bootstrap_ci: bool = True,
):
    xs = [math.log10(k) if use_log_x else float(k) for k in sizes]
    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    for method in methods:
        x_use, means, yerr_lo, yerr_hi = [], [], [], []
        for si, k in enumerate(sizes):
            qmap = per_query_store.get(method, {}).get(int(k), {})
            vals = list(qmap.values())
            if not vals:
                continue
            m, lo, hi = mean_ci95(vals, use_bootstrap=bootstrap_ci)
            x_use.append(xs[si])
            means.append(m)
            yerr_lo.append(m - lo)
            yerr_hi.append(hi - m)

        if not means:
            continue

        ax.errorbar(
            x_use, means,
            yerr=[yerr_lo, yerr_hi],
            marker="o",
            linewidth=2.0,
            capsize=3,
            label=method,
        )

    ax.set_ylabel("NDCG@10", fontsize=12)
    ax.set_xlabel("Candidate size (N)", fontsize=12)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in sizes], fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True, fontsize=10)

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    ensure_dir(os.path.dirname(out_pdf))
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print("[PLOT SAVED]", out_pdf)


def plot_grouped_boxplots_by_size(
    dataset: str,
    methods: list,
    sizes: list,
    per_query_store: dict,
    out_pdf: str,
):
    """
    Grouped boxplot:
      For each size (categorical x), draw one box per method side-by-side (no overlap),
      with a gap between size groups.
    """
    fig, ax = plt.subplots(figsize=(11.0, 5.2))

    n_methods = len(methods)
    n_sizes = len(sizes)

    # group centers at 0..n_sizes-1 (categorical)
    group_centers = list(range(n_sizes))

    # width of each box and the spread of a group
    # group_span controls how wide the 5 boxes spread within each size group
    group_span = 0.75
    box_w = min(0.12, group_span / max(n_methods, 1) * 0.9)

    # method offsets within each group (centered around group center)
    offsets = [(-group_span / 2) + (i + 0.5) * (group_span / n_methods) for i in range(n_methods)]

    # colors: use matplotlib default cycle (no explicit colors)
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", None)
    if not cycle:
        cycle = [None] * n_methods  # fallback

    legend_handles = []

    for mi, method in enumerate(methods):
        color = cycle[mi % len(cycle)]

        data = []
        positions = []

        for si, k in enumerate(sizes):
            qmap = per_query_store.get(method, {}).get(int(k), {})
            vals = [float(v) for v in qmap.values()]
            if not vals:
                # keep alignment: put empty list; matplotlib boxplot doesn't like empty,
                # so we skip this box if no data
                continue

            data.append(vals)
            positions.append(group_centers[si] + offsets[mi])

        if not data:
            continue

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=box_w,
            patch_artist=True,   # enables fill color
            showfliers=False,
            manage_ticks=False,
        )

        # style boxes for this method
        for patch in bp["boxes"]:
            if color is not None:
                patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_linewidth(1.2)

        for key in ["whiskers", "caps", "medians"]:
            for artist in bp[key]:
                artist.set_linewidth(1.2)

        # legend handle (one per method)
        if color is not None:
            h = ax.plot([], [], color=color, label=method, linewidth=6)[0]
        else:
            h = ax.plot([], [], label=method, linewidth=6)[0]
        legend_handles.append(h)

    ax.set_ylabel("NDCG@10", fontsize=12)
    ax.set_xlabel("Candidate size (N)", fontsize=12)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([str(k) for k in sizes], fontsize=11)

    ax.grid(True, axis="y", alpha=0.3)

    # legend outside
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True, fontsize=10)

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    ensure_dir(os.path.dirname(out_pdf))
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print("[PLOT SAVED]", out_pdf)


def plot_mean_lines_only(
    dataset: str,
    methods: list,
    sizes: list,
    per_query_store: dict,
    out_pdf: str,
    use_log_x: bool = True,
):
    xs = [math.log10(k) if use_log_x else float(k) for k in sizes]

    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    for method in methods:
        x_use, means = [], []
        for si, k in enumerate(sizes):
            qmap = per_query_store.get(method, {}).get(int(k), {})
            vals = list(qmap.values())
            if not vals:
                continue
            means.append(float(np.mean(vals)))
            x_use.append(xs[si])

        if not means:
            continue

        ax.plot(x_use, means, marker="o", linewidth=2.0, label=method)

    ax.set_ylabel("NDCG@10", fontsize=12)
    ax.set_xlabel("Candidate size (N)", fontsize=12)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in sizes], fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True, fontsize=10)

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    ensure_dir(os.path.dirname(out_pdf))
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print("[PLOT SAVED]", out_pdf)
 


# Example overrides (you can pass these in; sizes don't have to match these keys)
PASTEL_SAMPLE = {"N5":"#C7EA46", "N10":"#F6BDC7" , "N20":"#C1D9E9"}
DARK_SAMPLE   = {"N5":"#6E8B3D", "N10":"#A33A4B" , "N20":"#3B688A"}

# Default pastel/dark pairs (extendable) — used when you don't provide overrides
DEFAULT_PASTEL_DARK_PAIRS = [
    ("#F6BDC7", "#A33A4B"),  # pink
    ("#C1D9E9", "#3B688A"),  # blue
    ("#C7EA46", "#6E8B3D"),  # lime
    ("#F9E2AE", "#B07D26"),  # soft yellow
    ("#D7C7F4", "#5B3F8B"),  # lavender
    ("#BFE7D9", "#2F7C66"),  # mint
    ("#FFD6A5", "#B35C1E"),  # peach
    ("#CDE7BE", "#4A7B3C"),  # light green
    ("#F0CFEA", "#8A3B77"),  # magenta-ish
    ("#C9D1FF", "#3645A8"),  # periwinkle
]

def build_size_color_maps(
    sizes,
    pastel_override=None,
    dark_override=None,
    legend_labels=None,
):
    """
    Returns dicts:
      pastel_map[key] and dark_map[key] where key is a legend key like "N10".
    - sizes: [10,20,50,...]
    - legend_labels: optional list of legend strings per size (same length as sizes).
      If None, we use "N{size}".
    - override dicts are applied by key (e.g., {"N10":"#..."}).
    """
    if legend_labels is None:
        keys = [f"N{int(s)}" for s in sizes]
    else:
        # allow custom labels; still store by keys N{size} internally
        keys = [f"N{int(s)}" for s in sizes]

    pastel_map = {}
    dark_map = {}

    # base assignment from default pairs (cycled if more sizes than pairs)
    for i, key in enumerate(keys):
        p, d = DEFAULT_PASTEL_DARK_PAIRS[i % len(DEFAULT_PASTEL_DARK_PAIRS)]
        pastel_map[key] = p
        dark_map[key] = d

    # apply overrides (if provided)
    pastel_override = pastel_override or {}
    dark_override = dark_override or {}
    for k, v in pastel_override.items():
        pastel_map[k] = v
    for k, v in dark_override.items():
        dark_map[k] = v

    # legend labels for display
    if legend_labels is None:
        display_labels = {f"N{int(s)}": f"{int(s)}" for s in sizes}  # legend shows "10", "20", ...
    else:
        display_labels = {f"N{int(s)}": legend_labels[i] for i, s in enumerate(sizes)}

    return pastel_map, dark_map, display_labels


DISPLAY_NAME = {
    "MonoT5_220M": "MonoT5",      # <- remove 220 on plot
    "MonoT5_3B":   "MonoT5 (3B)",
    "GPT35":       "GPT-3.5",
    "GPT4oMini":   "GPT-4o-mini",
    "MonoBERT":    "MonoBERT",
    "BM25":        "BM25",
}

def pretty_method_name(m: str) -> str:
    return DISPLAY_NAME.get(m, m)



def plot_grouped_boxplots_by_model(
    dataset: str,
    methods: list,
    sizes: list,
    per_query_store: dict,
    out_pdf: str,
    # ---- labeling controls
    xlabel: str = "Model",
    ylabel: str = "NDCG@10",
    x_tick_fontsize: int = 17,
    y_tick_fontsize: int = 17,
    x_tick_rotation: int = 0,
    # ---- legend controls
    legend_title: str = "Size",
    legend_fontsize: int = 16,
    legend_title_fontsize: int = 14,
    legend_labels: list = None,  # optional custom legend labels per size
    # ---- color controls (optional overrides)
    pastel_override: dict = None,
    dark_override: dict = None,
):
    """
    Grouped boxplot (NO overlap):
      X-axis = models (methods)
      Within each model group: one box per size.
    Style:
      Fill color = pastel, edge/median/whiskers = dark (per size).
    """

    fig, ax = plt.subplots(figsize=(11.5, 5.2))

    n_methods = len(methods)
    n_sizes = len(sizes)

    # categorical x positions: one group per model
    group_centers = list(range(n_methods))

    # spacing inside each model group
    group_span = 0.82
    box_w = min(0.13, group_span / max(n_sizes, 1) * 0.9)
    offsets = [(-group_span / 2) + (i + 0.5) * (group_span / n_sizes) for i in range(n_sizes)]

    # build color maps for sizes
    pastel_map, dark_map, display_labels = build_size_color_maps(
        sizes=sizes,
        pastel_override=pastel_override,
        dark_override=dark_override,
        legend_labels=legend_labels,
    )
    
    legend_handles = []
    legend_labels_out = []


    # draw one "layer" per size (so each size has consistent color)
    for si, k in enumerate(sizes):
        size_key = f"N{int(k)}"
        face = pastel_map[size_key]
        edge = dark_map[size_key]

        data = []
        positions = []
        for mi, method in enumerate(methods):
            qmap = per_query_store.get(method, {}).get(int(k), {})
            vals = [float(v) for v in qmap.values()]
            if not vals:
                # skip missing
                continue
            data.append(vals)
            positions.append(group_centers[mi] + offsets[si])

        if not data:
            continue

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=box_w,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        
        legend_handles.append(
        Patch(facecolor=face, edgecolor=edge, linewidth=1.2, alpha=1)
    )
        legend_labels_out.append(display_labels[size_key])


        # box fill + edge
        for patch in bp["boxes"]:
            patch.set_facecolor(face)
            patch.set_edgecolor(edge)
            patch.set_alpha(1)
            patch.set_linewidth(1.6)

        # whiskers/caps/medians in dark edge color
        for w in bp["whiskers"]:
            w.set_color(edge)
            w.set_linewidth(1.4)
        for c in bp["caps"]:
            c.set_color(edge)
            c.set_linewidth(1.4)
        for m in bp["medians"]:
            m.set_color(edge)
            m.set_linewidth(1.8)

        # legend proxy for this size
        ax.plot([], [], color=edge, label=display_labels[size_key], linewidth=6)

    # labels & ticks
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([pretty_method_name(m) for m in methods],
                   fontsize=x_tick_fontsize, rotation=x_tick_rotation)
    
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])


    ax.tick_params(axis="y", labelsize=y_tick_fontsize)

    ax.grid(True, axis="y", alpha=0.3)

    # legend outside
    ax.legend(
        handles=legend_handles,
        labels=legend_labels_out,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=True,
        fontsize=legend_fontsize,
        title=legend_title,
        title_fontsize=legend_title_fontsize,
        handlelength=1.0,
        handleheight=0.7,
        borderpad=0.6,
        labelspacing=0.5,
        handletextpad=0.6,
    )


    fig.tight_layout(rect=[0, 0, 0.82, 1])
    ensure_dir(os.path.dirname(out_pdf))
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print("[PLOT SAVED]", out_pdf)

    
    


# ============================================================
# Plot-only helper (no reranking)
# ============================================================

def plot_only_for_datasets(datasets: List[str], methods: List[str], sizes: List[int]):
    
    run_tag = time.strftime("%Y%m%d_%H%M%S")

    
    for dataset in datasets:
        k0 = max(sizes) if sizes else 10
        bm25_res, qrels, topics, searcher = retrieve_topk(dataset, k=k0)
        qrels = normalize_qrels(qrels)

        maxg = max_grade_from_qrels(qrels)
        idcg_c = idcg_constant(maxg, k=TOP_K_EVAL)

        per_query_store: Dict[str, Dict[int, Dict[str, float]]] = {}
        for k in sizes:
            for method in methods:
                cache_key = method
#                 run_docids = load_cached_run_docids(dataset, k, cache_key)
                run_docids = load_cached_run_docids(dataset, k, method)


#                 if run_docids is None:
#                     if method == "GPT35":
#                         run_docids = load_cached_run_docids(dataset, k, "gpt-3.5-turbo")
#                     if method == "GPT4oMini":
#                         run_docids = load_cached_run_docids(dataset, k, "gpt-4o-mini")

#                 if run_docids is None:
#                     print(f"[SKIP] No cache for dataset={dataset} k={k} method={method}")
#                     continue

                run_docids = normalize_run_docids(run_docids)
                per_q = build_per_query_ndcg(run_docids, qrels, idcg_c, k=TOP_K_EVAL)
                per_query_store.setdefault(method, {})[int(k)] = per_q

        out_pdf = os.path.join(PLOTS_ROOT, dataset, f"{dataset}_plot_only_{time.strftime('%Y%m%d_%H%M%S')}.pdf")
#         plot_dataset_line_and_box(dataset, methods, sizes, per_query_store, out_pdf, use_log_x=True)

        # ------------------------------------------------------------
        # Generate ALL THREE plot types for this dataset
        # ------------------------------------------------------------
        band_pdf = os.path.join(
            PLOTS_ROOT, dataset,
            f"{dataset}_mean_ci_band_{run_tag}.pdf"
        )
        point_pdf = os.path.join(
            PLOTS_ROOT, dataset,
            f"{dataset}_mean_ci_points_{run_tag}.pdf"
        )
        box_pdf = os.path.join(
            PLOTS_ROOT, dataset,
            f"{dataset}_grouped_box_logx_{run_tag}.pdf"
        )
        
        mean_pdf = os.path.join(
                PLOTS_ROOT, dataset,
                f"{dataset}_mean_lines_only_{run_tag}.pdf"
            )

#         plot_mean_lines_with_ci_band(
#             dataset=dataset,
#             methods=methods,
#             sizes=sizes,
#             per_query_store=per_query_store,
#             out_pdf=band_pdf,
#             use_log_x=True,
#             bootstrap_ci=True,
#         )

#         plot_mean_lines_with_point_ci(
#             dataset=dataset,
#             methods=methods,
#             sizes=sizes,
#             per_query_store=per_query_store,
#             out_pdf=point_pdf,
#             use_log_x=True,
#             bootstrap_ci=True,
#         )
        
#         plot_mean_lines_only(
#             dataset=dataset,
#             methods=methods,
#             sizes=sizes,
#             per_query_store=per_query_store,
#             out_pdf=mean_pdf,
#             use_log_x=True,
#         )




#         plot_grouped_boxplots_by_size(
#             dataset=dataset,
#             methods=methods,
#             sizes=sizes,
#             per_query_store=per_query_store,
#             out_pdf=box_pdf,
#         )


        box_by_model_pdf = os.path.join(
            PLOTS_ROOT, dataset,
            f"{dataset}_grouped_box_by_model_{run_tag}.pdf"
        )


        
        plot_grouped_boxplots_by_model(
            dataset=dataset,
            methods=methods,
            sizes=sizes,
            per_query_store=per_query_store,
            out_pdf=box_by_model_pdf,

            # labels
            xlabel="Model",
            ylabel="NDCG@10",
            x_tick_fontsize=16,
            y_tick_fontsize=16,
            x_tick_rotation=0,

            # legend
            legend_title="Size",
            legend_fontsize=14,
            legend_title_fontsize=14,
            # legend_labels=["10 docs","20 docs","50 docs","100 docs","200 docs"],  # optional

            # color overrides (optional)
#             pastel_override=PASTEL_SAMPLE,
#             dark_override=DARK_SAMPLE,
        )





if __name__ == "__main__":
    
#     run_experiments()

#     ["dl19","trec-covid", "webis-touche2020", "trec-news",]
    plot_only_for_datasets(["dl19","trec-covid", "webis-touche2020", "trec-news",], METHODS, [10, 20, 50, 100, 200])
