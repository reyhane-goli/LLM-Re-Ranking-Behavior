#!/usr/bin/env python3
"""
run_selected_models_maxgrade.py

One standalone runner that:
  1) Retrieves BM25 top-K (default 100)
  2) Reranks + caches supervised rerankers (MonoBERT, MonoT5-220, MonoT5-3B)
  3) Loads existing LLM rerank logs for GPT-3.5-turbo and GPT-4o (NO API calls)
  4) Evaluates using *new* nDCG@10 where:
       IDCG@10 = DCG([max_grade] * 10)   (dataset-level constant),
     i.e., assume 10 docs all at maximum relevance grade.

Concurrency-safe:
  - Each run writes into: results/run_<run_id>/ and logs/run_<run_id>.txt
  - No overwriting across terminals.

Example (one dataset per terminal):
  python run_selected_models_maxgrade.py --datasets trec-covid --run-id trec-covid
"""

""" 
rel=0 → gain 0

rel=1 → gain 1

rel=2 → gain 3

rel=3 → gain 7

"""

"""
python run_selected_models_maxgrade.py --datasets dl19             --run-id dl19
python run_selected_models_maxgrade.py --datasets dl20             --run-id dl20
python run_selected_models_maxgrade.py --datasets trec-covid       --run-id trec-covid
python run_selected_models_maxgrade.py --datasets nfcorpus         --run-id nfcorpus
python run_selected_models_maxgrade.py --datasets webis-touche2020 --run-id webis-touche2020
python run_selected_models_maxgrade.py --datasets dbpedia-entity   --run-id dbpedia-entity
python run_selected_models_maxgrade.py --datasets scifact          --run-id scifact
python run_selected_models_maxgrade.py --datasets signal1m         --run-id signal1m
python run_selected_models_maxgrade.py --datasets trec-news        --run-id trec-news
python run_selected_models_maxgrade.py --datasets robust04         --run-id robust04


or 

python run_selected_models_maxgrade.py --datasets dl19 --run-id dl19 --compute-residuals


"""

import os
import re
import json
import math
import argparse
import traceback
from typing import Dict, List, Tuple, Optional

import pandas as pd

from config import RESULTS_DIR, LOGS_DIR, MONOBERT_CKPT, MONOT5_BASE_CKPT, MONOT5_3B_CKPT
from utils import TeeLogger, ts_filename, timestamp
from retrieve_bm25 import retrieve_topk
from rerank_supervised import MonoBERTReRanker, MonoT5ReRanker, rerank_with_progress


# ---------------------------
# Methods you care about
# ---------------------------
SUPPORTED_METHODS = ["bm25", "monobert", "monot5_220", "monot5_3b", "gpt35", "gpt4o"]

METHOD_LABEL = {
    "bm25": "BM25",
    "monobert": "MonoBERT",
    "monot5_220": "MonoT5 (220M)",
    "monot5_3b": "MonoT5 (3B)",
    "gpt35": "GPT-3.5-turbo",
    "gpt4o": "GPT-4o",
}

# If your log folders use slightly different model names, add more aliases here.
LLM_MODEL_DIR_ALIASES = {
    "gpt35": ["gpt-3.5-turbo"],
    "gpt4o": ["gpt-4o", "gpt4o"],
}


# ---------------------------
# Helpers: robust types
# ---------------------------
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _atomic_write_json(obj: dict, path: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def _to_str_keys(d: dict) -> dict:
    out = {}
    for k, v in (d or {}).items():
        out[str(k)] = v
    return out

def _normalize_topics(topics_raw) -> Dict[str, str]:
    """
    Convert Pyserini topics into: {qid(str): query_text(str)}.
    """
    topics = {}
    if isinstance(topics_raw, dict):
        for qid, q in topics_raw.items():
            qid = str(qid)
            if isinstance(q, dict):
                topics[qid] = str(q.get("title") or q.get("text") or q.get("query") or next(iter(q.values()), ""))
            else:
                topics[qid] = str(q)
    elif isinstance(topics_raw, list):
        for i, q in enumerate(topics_raw):
            qid = str(i)
            if isinstance(q, dict):
                topics[qid] = str(q.get("title") or q.get("text") or q.get("query") or next(iter(q.values()), ""))
            else:
                topics[qid] = str(q)
    return topics

def _intify_qrels(qrels: Dict) -> Dict[str, Dict[str, int]]:
    qrels2 = {}
    for qid, docs in (qrels or {}).items():
        qid = str(qid)
        qrels2[qid] = {}
        for docid, rel in (docs or {}).items():
            try:
                qrels2[qid][str(docid)] = int(float(rel))
            except Exception:
                qrels2[qid][str(docid)] = 0
    return qrels2

def max_relevance_grade(qrels: Dict[str, Dict[str, int]]) -> int:
    mg = 0
    for _qid, docs in qrels.items():
        for _docid, rel in docs.items():
            if rel > mg:
                mg = rel
    return int(mg)

# def _dcg_at_k(rels: List[int], k: int = 10) -> float:
#     s = 0.0
#     for i, rel in enumerate(rels[:k]):
#         if rel <= 0:
#             continue
#         s += float(rel) / math.log2(i + 2)
#     return s

# def ndcg10_maxgrade(run: Dict[str, Dict[str, float]], qrels: Dict[str, Dict[str, int]], max_grade: int, k: int = 10) -> float:
#     """
#     Mean nDCG@k with dataset-level constant IDCG = DCG([max_grade]*k)
#     """
#     if max_grade <= 0:
#         return 0.0
#     idcg = _dcg_at_k([max_grade] * k, k=k)
#     if idcg <= 0:
#         return 0.0

#     ndcgs = []
#     for qid, doc_scores in run.items():
#         qid = str(qid)
#         if qid not in qrels:
#             continue
#         ranked = sorted(doc_scores.items(), key=lambda x: (-float(x[1]), str(x[0])))
#         rels = [qrels[qid].get(str(docid), 0) for docid, _ in ranked[:k]]
#         dcg = _dcg_at_k(rels, k=k)
#         ndcgs.append(dcg / idcg)
#     return sum(ndcgs) / len(ndcgs) if ndcgs else 0.0


def _gain(rel: int) -> float:
    # trec_eval/pytrec_eval nDCG gain: 2^rel - 1
    rel = int(rel)
    return (2 ** rel) - 1.0

def _dcg_at_k(rels: List[int], k: int = 10) -> float:
    # trec_eval discount: log2(rank+1) with rank starting at 1
    s = 0.0
    for i, rel in enumerate(rels[:k]):  # i=0 => rank=1
        if rel <= 0:
            continue
        s += _gain(rel) / math.log2(i + 2)  # log2(rank+1)
    return s

def ndcg10_maxgrade(run: Dict[str, Dict[str, float]],
                    qrels: Dict[str, Dict[str, int]],
                    max_grade: int,
                    k: int = 10) -> float:
    """
    Mean nDCG@k with dataset-level constant IDCG = DCG([max_grade]*k),
    using trec_eval/pytrec_eval gain (2^rel - 1).
    """
    if max_grade <= 0:
        return 0.0

    idcg = _dcg_at_k([max_grade] * k, k=k)  # constant per dataset
    if idcg <= 0:
        return 0.0

    ndcgs = []
    for qid, doc_scores in run.items():
        qid = str(qid)
        if qid not in qrels:
            continue
        ranked = sorted(doc_scores.items(), key=lambda x: (-float(x[1]), str(x[0])))
        rels = [qrels[qid].get(str(docid), 0) for docid, _ in ranked[:k]]
        dcg = _dcg_at_k(rels, k=k)
        ndcgs.append(dcg / idcg)

    return sum(ndcgs) / len(ndcgs) if ndcgs else 0.0


def ndcg10_residual(run: Dict[str, Dict[str, float]],
                    qrels: Dict[str, Dict[str, int]],
                    max_grade: int,
                    k: int = 10,
                    unjudged_rel: int = -1) -> float:
    """
    Mean NDCG@k residual under the SAME global normalization as ndcg10_maxgrade().

    Residual = (extra DCG that *could* come from unjudged docs in top-k)
               / IDCG_global

    unjudged_rel:
      -1  => treat unjudged docs as max_grade (DEFAULT / worst-case upper bound)
      >=0 => treat unjudged docs as this relevance value (e.g., 1)
    """
    if max_grade <= 0:
        return 0.0

    idcg = _dcg_at_k([max_grade] * k, k=k)  # same constant denominator
    if idcg <= 0:
        return 0.0

    assumed_rel = max_grade if unjudged_rel is None or int(unjudged_rel) < 0 else int(unjudged_rel)
    if assumed_rel <= 0:
        # If you set assumed rel to 0, residual is always 0 by definition.
        return 0.0

    residuals = []
    for qid, doc_scores in run.items():
        qid = str(qid)
        if qid not in qrels:
            continue

        judged = qrels[qid]  # dict: docid -> rel
        ranked = sorted(doc_scores.items(), key=lambda x: (-float(x[1]), str(x[0])))

        extra_dcg = 0.0
        for i, (docid, _score) in enumerate(ranked[:k]):  # i=0 => rank=1
            docid = str(docid)
            if docid not in judged:  # <-- unjudged in this query
                extra_dcg += _gain(assumed_rel) / math.log2(i + 2)

        residuals.append(extra_dcg / idcg)

    return sum(residuals) / len(residuals) if residuals else 0.0


def ndcg10_maxgrade_with_residual(run: Dict[str, Dict[str, float]],
                                 qrels: Dict[str, Dict[str, int]],
                                 max_grade: int,
                                 k: int = 10,
                                 unjudged_rel: int = -1) -> Tuple[float, float]:
    """
    Convenience: returns (ndcg10_maxgrade, ndcg10_residual).
    """
    nd = ndcg10_maxgrade(run, qrels, max_grade=max_grade, k=k)
    res = ndcg10_residual(run, qrels, max_grade=max_grade, k=k, unjudged_rel=unjudged_rel)
    return nd, res


def bm25_results_to_run(bm25_res: Dict[str, List[Tuple[str, float]]]) -> Dict[str, Dict[str, float]]:
    run = {}
    for qid, items in (bm25_res or {}).items():
        qid = str(qid)
        run[qid] = {str(docid): float(score) for docid, score in items}
    return run


# ---------------------------
# Caching: supervised rerankers
# ---------------------------
def cache_dir(cache_root: str, dataset: str, method: str, depth: int) -> str:
    return os.path.join(cache_root, dataset, f"{method}_top{depth}")

def save_rerank_cache(
    cache_root: str,
    dataset: str,
    method: str,
    depth: int,
    topics: Dict[str, str],
    bm25_res: Dict[str, List[Tuple[str, float]]],
    reranked: Dict[str, List[Tuple[str, float]]],
    skip_if_exists: bool = True,
) -> None:
    out_dir = cache_dir(cache_root, dataset, method, depth)
    _ensure_dir(out_dir)
    for qid, query in topics.items():
        path = os.path.join(out_dir, f"qid_{qid}.json")
        if skip_if_exists and os.path.exists(path):
            continue
        inp = [{"docid": str(d), "score": float(s)} for d, s in bm25_res.get(qid, [])]
        out = [{"docid": str(d), "score": float(s)} for d, s in reranked.get(qid, [])]
        _atomic_write_json(
            {
                "dataset": dataset,
                "method": method,
                "bm25_depth": int(depth),
                "qid": str(qid),
                "query": str(query),
                "input_docs": inp,
                "output_docs": out,
                "input_docids": [x["docid"] for x in inp],
                "output_docids": [x["docid"] for x in out],
                "timestamp": timestamp(),
            },
            path,
        )

def load_cached_run(cache_root: str, dataset: str, method: str, depth: int) -> Optional[Dict[str, Dict[str, float]]]:
    out_dir = cache_dir(cache_root, dataset, method, depth)
    if not os.path.isdir(out_dir):
        return None
    run = {}
    for fn in os.listdir(out_dir):
        if not fn.endswith(".json"):
            continue
        p = os.path.join(out_dir, fn)
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            qid = str(obj.get("qid"))
            out_docs = obj.get("output_docs") or []
            if out_docs and isinstance(out_docs[0], dict) and "score" in out_docs[0]:
                run[qid] = {str(d["docid"]): float(d.get("score", 0.0)) for d in out_docs}
            else:
                docids = obj.get("output_docids") or []
                run[qid] = {str(d): float(len(docids) - i) for i, d in enumerate(docids)}
        except Exception:
            continue
    return run if run else None


# ---------------------------
# Loading LLM logs (existing)
# ---------------------------
def find_llm_model_dir(llm_root: str, dataset: str, method: str) -> str:
    # Your structure is: logs_llm_rerank/<dataset>/gpt-3.5-turbo and gpt-4o
    model_dir = {
        "gpt35": "gpt-3.5-turbo",
        "gpt4o": "gpt-4o",
    }.get(method)

    if model_dir is None:
        raise ValueError(f"Unknown LLM method: {method}")

    cand = os.path.join(llm_root, dataset, model_dir)
    if os.path.isdir(cand):
        return cand

    raise FileNotFoundError(
        f"Could not find LLM logs for dataset='{dataset}' in '{llm_root}'. "
        f"Expected folder: {cand}"
    )


def load_llm_run(llm_root: str, dataset: str, method: str) -> Dict[str, Dict[str, float]]:
    base = find_llm_model_dir(llm_root, dataset, method)
    run = {}

    json_files = [f for f in os.listdir(base) if f.endswith(".json")]
    for fn in json_files:
        path = os.path.join(base, fn)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        qid = obj.get("qid")
        if qid is None:
            # fallback: try parse from filename
            m = re.search(r"qid[_-]?(\d+)", fn)
            qid = m.group(1) if m else fn.replace(".json", "")
        qid = str(qid)

        # Prefer output_docs with scores; otherwise output_docids order
        out_docs = obj.get("output_docs") or []
        if out_docs and isinstance(out_docs[0], dict) and "score" in out_docs[0]:
            run[qid] = {str(d["docid"]): float(d.get("score", 0.0)) for d in out_docs}
        else:
            docids = obj.get("output_docids") or []
            run[qid] = {str(d): float(len(docids) - i) for i, d in enumerate(docids)}

    return run


# ---------------------------
# Document fetcher for supervised rerank
# ---------------------------
def make_doc_fetcher(searcher):
    from rerank_supervised import _get_text_from_raw
    def fetch(docid: str) -> str:
        doc = searcher.doc(docid)
        if doc is None:
            return ""
        return _get_text_from_raw(doc.raw())
    return fetch


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True, help="e.g., dl19 dl20 trec-covid nfcorpus ...")
    ap.add_argument("--methods", nargs="+", default=SUPPORTED_METHODS, help=f"Subset of: {SUPPORTED_METHODS}")
    ap.add_argument("--bm25-depth", type=int, default=100, help="BM25 candidate depth (default 100).")
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size for MonoBERT / MonoT5 scoring.")
    ap.add_argument("--cache-root", type=str, default="rerank_cache", help="Where to store per-qid rerank caches.")
    ap.add_argument("--llm-log-root", type=str, default="logs_llm_rerank", help="Existing GPT logs root.")
    ap.add_argument("--run-id", type=str, default=None, help="Unique run id. Default: timestamp_pid")
    ap.add_argument("--force-rerank", action="store_true", help="Ignore caches and rerun supervised rerankers.")
    ap.add_argument("--compute-residuals", action="store_true",
                help="Also compute NDCG@10 residuals from unjudged docs in top-10.")
    ap.add_argument("--unjudged-rel", type=int, default=-1,
                help="Assumed relevance for unjudged docs when computing residuals. "
                     "-1 (default) means use dataset max_grade; set 1 to assume rel=1.")

    args = ap.parse_args()

    methods = [m.strip().lower() for m in args.methods]
    bad = [m for m in methods if m not in SUPPORTED_METHODS]
    if bad:
        raise ValueError(f"Unknown methods: {bad}. Allowed: {SUPPORTED_METHODS}")

    run_id = args.run_id or f"{ts_filename()}_{os.getpid()}"

    _ensure_dir(RESULTS_DIR)
    _ensure_dir(LOGS_DIR)
    _ensure_dir(args.cache_root)

    # Run-specific output folder (prevents overwriting)
    out_dir = os.path.join(RESULTS_DIR, f"run_{run_id}")
    _ensure_dir(out_dir)

    log_path = os.path.join(LOGS_DIR, f"run_{run_id}.txt")
    tee = TeeLogger(log_path)

    print(f"[INFO {timestamp()}] run_id={run_id}")
    print(f"[INFO {timestamp()}] out_dir={out_dir}")
    print(f"[INFO {timestamp()}] log={log_path}")

    rows = []
    max_grade_map = {}

    for dataset in args.datasets:
        print(f"\n==============================")
        print(f"DATASET: {dataset}")
        print(f"==============================")

        try:
            bm25_res, qrels_raw, topics_raw, searcher = retrieve_topk(dataset, k=args.bm25_depth)
            bm25_res = _to_str_keys(bm25_res)
            topics = _normalize_topics(topics_raw)
            qrels = _intify_qrels(qrels_raw)

            mg = max_relevance_grade(qrels)
            max_grade_map[dataset] = mg
            print(f"[INFO] Max relevance grade for {dataset} = {mg}")

            # BM25 evaluation
            if "bm25" in methods:
                bm25_run = bm25_results_to_run(bm25_res)
                label = METHOD_LABEL["bm25"]

                if args.compute_residuals:
                    nd, res = ndcg10_maxgrade_with_residual(
                        bm25_run, qrels, max_grade=mg, k=10,
                        unjudged_rel=args.unjudged_rel
                    )
                    rows.append((dataset, label, nd, res))
                    print(f"[RESULT] {dataset} | {label} | nDCG@10(max-grade) = {nd:.4f} | residual = {res:.4f}")
                else:
                    nd = ndcg10_maxgrade(bm25_run, qrels, max_grade=mg, k=10)
                    rows.append((dataset, label, nd))
                    print(f"[RESULT] {dataset} | {label} | nDCG@10(max-grade) = {nd:.4f}")



                # cache BM25 as "output = input"
                save_rerank_cache(
                    cache_root=args.cache_root,
                    dataset=dataset,
                    method="bm25",
                    depth=args.bm25_depth,
                    topics=topics,
                    bm25_res=bm25_res,
                    reranked=bm25_res,
                    skip_if_exists=True,
                )

            doc_fetch = make_doc_fetcher(searcher)

            # Supervised rerankers (cache or rerun)
            for m in ["monobert", "monot5_220", "monot5_3b"]:
                if m not in methods:
                    continue
                label = METHOD_LABEL[m]

                cached = None if args.force_rerank else load_cached_run(args.cache_root, dataset, m, args.bm25_depth)
                if cached is not None:
                    if args.compute_residuals:
                        nd, res = ndcg10_maxgrade_with_residual(
                            cached, qrels, max_grade=mg, k=10,
                            unjudged_rel=args.unjudged_rel
                        )
                        rows.append((dataset, label, nd, res))
                        print(f"[CACHE] {dataset} | {label} loaded")
                        print(f"[RESULT] {dataset} | {label} | nDCG@10(max-grade) = {nd:.4f} | residual = {res:.4f}")
                    else:
                        nd = ndcg10_maxgrade(cached, qrels, max_grade=mg, k=10)
                        rows.append((dataset, label, nd))
                        print(f"[CACHE] {dataset} | {label} loaded")
                        print(f"[RESULT] {dataset} | {label} | nDCG@10(max-grade) = {nd:.4f}")
                    continue


                print(f"[INFO] Running supervised rerank: {label}")
                if m == "monobert":
                    rr_model = MonoBERTReRanker(MONOBERT_CKPT, max_len=512)
                    rr = rerank_with_progress(label, rr_model, bm25_res, topics, doc_fetch,
                                              batch_size=args.batch_size, colour="blue")
                elif m == "monot5_220":
                    rr_model = MonoT5ReRanker(MONOT5_BASE_CKPT, max_len=512)
                    rr = rerank_with_progress(label, rr_model, bm25_res, topics, doc_fetch,
                                              batch_size=max(1, args.batch_size // 2), colour="blue")
                else:  # monot5_3b
                    rr_model = MonoT5ReRanker(MONOT5_3B_CKPT, max_len=512)
                    rr = rerank_with_progress(label, rr_model, bm25_res, topics, doc_fetch,
                                              batch_size=max(1, args.batch_size // 4), colour="blue")

                rr_run = bm25_results_to_run(rr)

                if args.compute_residuals:
                    nd, res = ndcg10_maxgrade_with_residual(
                        rr_run, qrels, max_grade=mg, k=10,
                        unjudged_rel=args.unjudged_rel
                    )
                    rows.append((dataset, label, nd, res))
                    print(f"[RESULT] {dataset} | {label} | nDCG@10(max-grade) = {nd:.4f} | residual = {res:.4f}")
                else:
                    nd = ndcg10_maxgrade(rr_run, qrels, max_grade=mg, k=10)
                    rows.append((dataset, label, nd))
                    print(f"[RESULT] {dataset} | {label} | nDCG@10(max-grade) = {nd:.4f}")


                save_rerank_cache(
                    cache_root=args.cache_root,
                    dataset=dataset,
                    method=m,
                    depth=args.bm25_depth,
                    topics=topics,
                    bm25_res=bm25_res,
                    reranked=rr,
                    skip_if_exists=True,
                )

            # LLM (load existing logs; default is whatever you already logged e.g., top-30)
            for m in ["gpt35", "gpt4o"]:
                if m not in methods:
                    continue
                label = METHOD_LABEL[m]
                print(f"[INFO] Loading existing LLM logs: {label}")

                llm_run = load_llm_run(args.llm_log_root, dataset, m)

                if args.compute_residuals:
                    nd, res = ndcg10_maxgrade_with_residual(
                        llm_run, qrels, max_grade=mg, k=10,
                        unjudged_rel=args.unjudged_rel
                    )
                    rows.append((dataset, label, nd, res))
                    print(f"[RESULT] {dataset} | {label} | nDCG@10(max-grade) = {nd:.4f} | residual = {res:.4f}")
                else:
                    nd = ndcg10_maxgrade(llm_run, qrels, max_grade=mg, k=10)
                    rows.append((dataset, label, nd))
                    print(f"[RESULT] {dataset} | {label} | nDCG@10(max-grade) = {nd:.4f}")


        except Exception as e:
            print(f"[ERROR] Dataset failed: {dataset} | {e}")
            traceback.print_exc()

    # Save outputs (run-specific)
    df_long = pd.DataFrame(rows, columns=["Dataset", "Method", "nDCG@10_maxgrade"])
    df_long_path = os.path.join(out_dir, "results_long.csv")
    df_long.to_csv(df_long_path, index=False)

    pivot = df_long.pivot_table(index="Method", columns="Dataset", values="nDCG@10_maxgrade", aggfunc="mean")
    pivot_path = os.path.join(out_dir, "results_wide.csv")
    pivot.to_csv(pivot_path)

    # Save max grade per dataset
    mg_path = os.path.join(out_dir, "max_grade_per_dataset.json")
    _atomic_write_json(max_grade_map, mg_path)

    print("\n==============================")
    print("Max relevance grade per dataset")
    print("==============================")
    for ds, g in max_grade_map.items():
        print(f"  - {ds}: {g}")

    print("\n==============================")
    print("Summary (nDCG@10 max-grade)")
    print("==============================")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(pivot.fillna("-"))

    print(f"\n[SAVED] {df_long_path}")
    print(f"[SAVED] {pivot_path}")
    print(f"[SAVED] {mg_path}")
    print(f"[LOG]   {log_path}")

    tee.close()


if __name__ == "__main__":
    main()
