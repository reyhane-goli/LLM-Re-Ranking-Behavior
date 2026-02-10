# ===============================================================
#  RQ3: Input Order Experiment (Two-Part Version)
# ===============================================================
# Part 1: generate_order_logs() → build new orderings & save GPT outputs
# Part 2: compute_order_metrics() → compute τ, RBO@0.79, nDCG@10 summaries
#
# Each dataset × model combination:
#   results/{dataset}_{model}/order_exp/{ordering}/{qid}/inXX_runY.json
#   results/{dataset}_{model}/results_order_summary.txt
# ===============================================================

import os
import json
from math import log2
import numpy as np
from scipy.stats import kendalltau
from rbo import RankingSimilarity
from tqdm import tqdm
import pandas as pd
from typing import List, Dict, Tuple

from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels

# Provided externally
from rank_gpt import create_permutation_instruction, run_llm, receive_permutation

# ───────────────────────────────
# Global settings
# ───────────────────────────────
DATASETS = ["trec-covid", "nfcorpus"]
MODELS   = ["gpt-3.5", "gpt-4o-mini"]

# DATASETS = ["nfcorpus"]
# MODELS   = ["gpt-3.5"]

ORDERINGS = ["rel_top", "rel_bottom", "rel_middle", "rel_ends"]
BASE_MODE = "half_relevant"
K = 10
NUM_INPUTS = 50
DETER_RUNS = 2
RBO_P = 0.79  # fixed for k=10

API_KEY = ""

# ───────────────────────────────
# Helper functions
# ───────────────────────────────
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def dcg_at_k(rels, k=10):
    return sum(rel / log2(i + 2) for i, rel in enumerate(rels[:k]))

def ndcg_for_out(out_docs, qrel_dict, k=10):
    labels = [int(qrel_dict.get(d, 0)) for d in out_docs[:k]]
    dcg = dcg_at_k(labels, k)
    ideal = sorted(labels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0

def rbo_score(a, b, p=0.79):
    return RankingSimilarity(a, b).rbo(p=p)

def ds_config(dataset: str) -> Tuple[str, str, str]:
    if dataset == "trec-covid":
        return ("beir-v1.0.0-trec-covid.flat", "beir-v1.0.0-trec-covid-test", "trec-covid")
    elif dataset == "nfcorpus":
        return ("beir-v1.0.0-nfcorpus.flat", "beir-v1.0.0-nfcorpus-test", "nfcorpus")
    else:
        raise ValueError("Unknown dataset")

def split_rel_nonrel_preserve(docids: List[str], qrel_dict: Dict[str, int]):
    rel, non = [], []
    for d in docids:
        if int(qrel_dict.get(d, 0)) > 0:
            rel.append(d)
        else:
            non.append(d)
    return rel, non

def make_ordering(docids: List[str], qrel_dict: Dict[str, int], ordering: str):
    rel, non = split_rel_nonrel_preserve(docids, qrel_dict)
    if not rel or not non:
        return list(docids)
    if ordering == "rel_top":
        return rel + non
    elif ordering == "rel_bottom":
        return non + rel
    elif ordering == "rel_middle":
        mid = len(non) // 2
        return non[:mid] + rel + non[mid:]
    elif ordering == "rel_ends":
        left_rel = len(rel) // 2
        return rel[:left_rel] + non + rel[left_rel:]
    return list(docids)

def read_baseline_inputs(base_dir: str, qid: str, num_inputs: int):
    qdir = os.path.join(base_dir, "N10", BASE_MODE, str(qid))
    inputs = []
    for i in range(num_inputs):
        f0 = os.path.join(qdir, f"in{i:02d}_run0.json")
        if not os.path.exists(f0):
            inputs.append([])
            continue
        try:
            log0 = json.load(open(f0))
            docids = log0["input_docids"]
            inputs.append(docids[:K])
        except Exception:
            inputs.append([])
    return inputs


def get_qrel_dict_robust(qrels_raw, qid, warn_if_missing=False, tag="qrels"):
    """
    Return the per-qid qrels dict, handling int/str key mismatches and
    numpy/int64 types. Tries exact, int(qid), str(qid). Optionally warns.
    """
    # exact (covers already-matching int or str keys)
    if qid in qrels_raw:
        return qrels_raw[qid]

    # try to coerce to int (works for '1', np.int64(1), etc.)
    try:
        qi = int(str(qid).strip())
        if qi in qrels_raw:
            return qrels_raw[qi]
    except Exception:
        pass

    # fallback to clean string key
    qs = str(qid).strip()
    if qs in qrels_raw:
        return qrels_raw[qs]

    if warn_if_missing:
        print(f"⚠️  [{tag}] No qrels found for qid={qid!r} (keys like: {list(qrels_raw)[:5]} ...)")
    return {}

def is_rel(docid, qrel_dict):
    try:
        return int(qrel_dict.get(docid, 0)) > 0
    except Exception:
        return False

def rel_nonrel_mask(docids, qrel_dict):
    # True for relevant, False for non-relevant
    return [is_rel(d, qrel_dict) for d in docids]

def count_rel_nonrel(docids, qrel_dict):
    m = rel_nonrel_mask(docids, qrel_dict)
    r = sum(1 for x in m if x)
    return r, len(m) - r

def validate_ordering(ordering_name, docids, qrel_dict):
    """
    Return (ok, reason) indicating whether 'docids' satisfy the intended pattern.
    Definitions used here (tight but practical):
      - rel_top:      all relevant first, then all non-relevant (no interleaving)
      - rel_bottom:   all non-relevant first, then all relevant
      - rel_middle:   non-relevant block(s) at the ends with a single contiguous relevant block strictly in the middle
      - rel_ends:     a (possibly empty) relevant prefix, then non-relevant block, then a (possibly empty) relevant suffix
                      BUT at least one relevant must exist in prefix or suffix (so rels at “ends”, not only middle).
    """
    m = rel_nonrel_mask(docids, qrel_dict)
    n = len(m)

    if n == 0:
        return False, "empty list"

    # utility to check "once it flips, it never flips back"
    def monotone_prefix_then_suffix(bits, first_val=True):
        flipped = False
        for b in bits:
            if not flipped:
                if b != first_val:
                    flipped = True
            else:
                if b == first_val:
                    return False
        return True

    if ordering_name == "rel_top":
        ok = monotone_prefix_then_suffix(m, first_val=True)
        return (ok, "" if ok else "interleaved (expected all rel then all nonrel)")

    if ordering_name == "rel_bottom":
        ok = monotone_prefix_then_suffix([not x for x in m], first_val=True)
        return (ok, "" if ok else "interleaved (expected all nonrel then all rel)")

    if ordering_name == "rel_middle":
        # pattern: nonrel* , rel+ , nonrel*
        i, j = 0, n - 1
        while i < n and (not m[i]):  # nonrel prefix
            i += 1
        while j >= 0 and (not m[j]): # nonrel suffix
            j -= 1
        if i > j:
            return False, "no relevant block in the middle"
        # middle must be all relevant
        ok_mid = all(m[k] for k in range(i, j + 1))
        ok = (i > 0 or j < n - 1) and ok_mid
        return (ok, "" if ok else "middle block not all relevant (or rels not centered)")

    if ordering_name == "rel_ends":
        # pattern: rel* , nonrel+ , rel*   with at least one rel in prefix or suffix
        i, j = 0, n - 1
        # prefix rel*
        while i < n and m[i]:
            i += 1
        # suffix rel*
        while j >= 0 and m[j]:
            j -= 1
        if i == 0 and j == n - 1:
            return False, "no rel at ends (all nonrel)"
        # middle (i..j) must be all nonrel (or empty if i>j)
        if i <= j:
            ok_mid = all((not m[k]) for k in range(i, j + 1))
        else:
            ok_mid = True
        ok = ok_mid and any(m)  # at least one rel exists
        return (ok, "" if ok else "nonrel not centered or missing rel ends")

    # default: accept
    return True, ""

def model_api_name(model: str) -> str:
    """Name to call in the API."""
    m = model.strip().lower()
    if m.startswith("gpt-3.5"):
        return "gpt-3.5-turbo"
    return model

# ───────────────────────────────
# PART 1: GENERATION
# ───────────────────────────────
def generate_order_logs():
    if API_KEY == "REPLACE_WITH_YOUR_KEY":
        raise RuntimeError("Please set OPENAI_API_KEY.")

    for dataset in DATASETS:
        INDEX_NAME, TOPICS_NAME, DATASET_SHORT = ds_config(dataset)
        topics = get_topics(TOPICS_NAME)
        qrels = get_qrels(TOPICS_NAME)

        searcher = LuceneSearcher.from_prebuilt_index(INDEX_NAME)

        for model in MODELS:
            results_root = f"results/{DATASET_SHORT}_{model}"
            base_dir = os.path.join(results_root, "N10", BASE_MODE)
            if not os.path.isdir(base_dir):
                print(f"⚠️  Skipping {dataset}×{model}: baseline {base_dir} missing.")
                continue

            qids = sorted(os.listdir(base_dir))
            print(f"\n=== Generating new orderings for {dataset} × {model} ===")

            for qid in tqdm(qids, desc=f"{dataset}_{model} generation"):
                # normalize topic key type
                qid_key = int(qid) if any(isinstance(k, int) for k in topics.keys()) else str(qid)
                if qid_key not in topics:
                    print(f"⚠️  Skipping {qid}: not found in topics ({list(topics.keys())[:5]}...)")
                    continue
#                 qrel_dict = qrels.get(str(qid), {})  # qrels always string-keyed
                qrel_dict = get_qrel_dict_robust(qrels, qid, warn_if_missing=True, tag="gen-qrels")

                query_text = topics[qid_key].get("title", "")

                baseline_inputs = read_baseline_inputs(results_root, qid, NUM_INPUTS)

#                 for inp_idx, docids in enumerate(baseline_inputs):
#                     if not docids:
#                         continue

#                     for ord_name in ORDERINGS:
#                         new_order = make_ordering(docids, qrel_dict, ord_name)
#                         hits = []
#                         for d in new_order:
#                             dobj = searcher.doc(d)
#                             if dobj:
#                                 hits.append({"docid": d, "content": dobj.raw()})
#                         if len(hits) < 2:
#                             continue

#                         out_dir = os.path.join(results_root, "order_exp", ord_name, str(qid))
#                         ensure_dir(out_dir)

#                         for run_idx in range(DETER_RUNS):
#                             item = {"query": query_text, "hits": [h.copy() for h in hits]}
#                             msgs = create_permutation_instruction(item, 0, len(hits), model_name=model)
#                             resp = run_llm(msgs, api_key=API_KEY, model_name=model)
#                             out  = receive_permutation(item, resp, 0, len(hits))
#                             log = {
#                                 "qid": qid,
#                                 "ordering": ord_name,
#                                 "size": K,
#                                 "input_index": inp_idx,
#                                 "run": run_idx,
#                                 "input_docids": [h["docid"] for h in hits],
#                                 "output_docids": [h["docid"] for h in out["hits"][:len(hits)]],
#                                 "response": resp
#                             }
#                             with open(os.path.join(out_dir, f"in{inp_idx:02d}_run{run_idx}.json"), "w") as f:
#                                 json.dump(log, f, indent=2)

                for inp_idx, docids in enumerate(baseline_inputs):
                    if not docids:
                        continue

                    for ord_name in ORDERINGS:
                        new_order = make_ordering(docids, qrel_dict, ord_name)

                        # ── Preflight: count rel/nonrel + ordering validity ──
                        rel_cnt, non_cnt = count_rel_nonrel(new_order, qrel_dict)
                        ok, reason = validate_ordering(ord_name, new_order, qrel_dict)

                        # Expectation (you can tighten these if you want exact counts):
                        # - rel_top / rel_bottom / rel_middle / rel_ends: just enforce pattern (ok == True).
                        if not ok:
                            print(f"[BAD {ord_name}] Q{qid} in={inp_idx:02d}  rel={rel_cnt} nonrel={non_cnt}  :: {reason}")
                            # Skip LLM for this bad input (so you don’t waste calls)
                            continue

                        # Build hits
                        hits = []
                        for d in new_order:
                            dobj = searcher.doc(d)
                            if dobj:
                                hits.append({"docid": d, "content": dobj.raw()})
                        if len(hits) < 2:
                            print(f"[SKIP {ord_name}] Q{qid} in={inp_idx:02d} :: too few retrievable hits ({len(hits)})")
                            continue

                        # ── LLM calls only after passing checks ──
                        out_dir = os.path.join(results_root, "order_exp", ord_name, str(qid))
                        ensure_dir(out_dir)
                        
                        api_model = model_api_name(model)

                        for run_idx in range(DETER_RUNS):
                            item = {"query": query_text, "hits": [h.copy() for h in hits]}
                            msgs = create_permutation_instruction(item, 0, len(hits), model_name=api_model)
                            resp = run_llm(msgs, api_key=API_KEY, model_name=api_model)
                            out  = receive_permutation(item, resp, 0, len(hits))
                            log = {
                                "qid": qid,
                                "ordering": ord_name,
                                "size": K,
                                "input_index": inp_idx,
                                "run": run_idx,
                                "input_docids": [h["docid"] for h in hits],
                                "output_docids": [h["docid"] for h in out["hits"][:len(hits)]],
                                "response": resp
                            }
                            with open(os.path.join(out_dir, f"in{inp_idx:02d}_run{run_idx}.json"), "w") as f:
                                json.dump(log, f, indent=2)

    print("\n✅ All order logs generated and saved.")

# ───────────────────────────────
# PART 2: ANALYSIS
# ───────────────────────────────


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# RQ3: Input Order Experiment — analysis-only safe version.

# Layout expected:
#   results/{dataset}_{model}/N10/half_relevant/{qid}/inXX_run{0,1}.json      # "shuffle" (baseline)
#   results/{dataset}_{model}/order_exp/{rel_top|rel_bottom|rel_middle|rel_ends}/{qid}/inXX_run{0,1}.json

# Each JSON must contain at least:
#   { "qid": ..., "input_index": i, "input_docids": [...], "output_docids": [...] }

# This script:
#   - Computes Kendall's τ, RBO@0.79, nDCG@10 for each ordering
#   - Aggregates by ordering and writes results_order_summary.txt
# """

# import os, json
# from math import log2
# from collections import OrderedDict
# from typing import Dict, List, Tuple

# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from scipy.stats import kendalltau
# from rbo import RankingSimilarity
# from pyserini.search import get_qrels

# # =========================
# # CONFIG
# # =========================
# DATASETS = ["nfcorpus"]          # or ["trec-covid", "nfcorpus"]
# MODELS   = ["gpt-4o-mini"]       # or ["gpt-3.5", "gpt-4o-mini"]
# BASE_MODE = "half_relevant"      # fixed per your setup
# K = 10                           # single size
# NUM_INPUTS = 50
# RBO_P = 0.79                     # fixed p for k=10
# ORDERINGS = ["shuffle", "rel_top", "rel_bottom", "rel_middle", "rel_ends"]

# def ds_config(dataset: str) -> Tuple[str, str, str]:
#     if dataset == "trec-covid":
#         return ("beir-v1.0.0-trec-covid.flat", "beir-v1.0.0-trec-covid-test", "trec-covid")
#     elif dataset == "nfcorpus":
#         return ("beir-v1.0.0-nfcorpus.flat", "beir-v1.0.0-nfcorpus-test", "nfcorpus")
#     else:
#         raise ValueError("Unknown dataset name")

# # =========================
# # HELPERS
# # =========================
# def _load_json(path: str):
#     try:
#         with open(path, "r") as f:
#             return json.load(f)
#     except Exception:
#         return None

# def get_qrel_dict_robust(qrels_raw, qid, warn_if_missing=False, tag="qrels"):
#     """Return per-qid qrels, handling int/str key mismatches."""
#     if qid in qrels_raw:
#         return qrels_raw[qid]
#     try:
#         qi = int(str(qid).strip())
#         if qi in qrels_raw:
#             return qrels_raw[qi]
#     except Exception:
#         pass
#     qs = str(qid).strip()
#     if qs in qrels_raw:
#         return qrels_raw[qs]
#     if warn_if_missing:
#         print(f"⚠️  [{tag}] missing qrels for qid={qid!r}")
#     return {}

# def dcg_at_k(rels: List[int], k: int = 10) -> float:
#     return sum(rel / log2(i + 2) for i, rel in enumerate(rels[:k]))

# def ndcg_for_out(out_docids: List[str], qrel_dict: Dict[str, int], k: int = 10) -> float:
#     labels = [int(qrel_dict.get(d, 0)) for d in out_docids[:k]]
#     dcg = dcg_at_k(labels, k)
#     ideal = sorted(labels, reverse=True)
#     idcg = dcg_at_k(ideal, k)
#     return (dcg / idcg) if idcg > 0 else 0.0

# def _dedup_preserve(seq: List[str]) -> List[str]:
#     return list(OrderedDict.fromkeys(seq))

# def rbo_score(a_ids: List[str], b_ids: List[str], p: float) -> float:
#     """RBO impl asserts unique; guard with dedup."""
#     a = _dedup_preserve(a_ids or [])
#     b = _dedup_preserve(b_ids or [])
#     if not a or not b:
#         return float("nan")
#     return RankingSimilarity(a, b).rbo(p=p)

# def kendall_tau_on_ids(a_ids: List[str], b_ids: List[str]) -> float:
#     """
#     Compute Kendall's τ between two permutations of the SAME id set.
#     Assumes lengths equal and same items (enforced by _sanitize_output).
#     """
#     if not a_ids or not b_ids or len(a_ids) != len(b_ids):
#         return float("nan")
#     # map id -> position in a
#     pos_a = {d: i for i, d in enumerate(a_ids)}
#     # sequence of positions of b in a
#     seq_in_a = [pos_a[d] for d in b_ids if d in pos_a]
#     if len(seq_in_a) < 2:
#         return float("nan")
#     # compare against perfectly sorted sequence (0..n-1)
#     tau, _ = kendalltau(seq_in_a, list(range(len(seq_in_a))))
#     return float(tau) if tau is not None else float("nan")

# def _sanitize_output(out_ids: List[str], input_ids: List[str], k: int) -> List[str]:
#     """
#     Enforce: outputs are a permutation (no dup), subset of inputs, length K.
#     If model returns fewer than K after filtering, pad with remaining inputs in input order.
#     """
#     input_set = set(input_ids)
#     seen = set()
#     filtered = []
#     for d in out_ids or []:
#         if d in input_set and d not in seen:
#             filtered.append(d)
#             seen.add(d)
#         if len(filtered) == k:
#             break
#     if len(filtered) < k:
#         for d in input_ids:
#             if d not in seen:
#                 filtered.append(d)
#                 seen.add(d)
#             if len(filtered) == k:
#                 break
#     return filtered[:k]

# def _paths_for_ordering(results_root: str):
#     """Return dict ordering -> folder path."""
#     return {
#         "shuffle":   os.path.join(results_root, "N10", BASE_MODE),  # baseline logs
#         "rel_top":   os.path.join(results_root, "order_exp", "rel_top"),
#         "rel_bottom":os.path.join(results_root, "order_exp", "rel_bottom"),
#         "rel_middle":os.path.join(results_root, "order_exp", "rel_middle"),
#         "rel_ends":  os.path.join(results_root, "order_exp", "rel_ends"),
#     }

# # =========================
# # ANALYSIS
# # =========================
# def compute_order_metrics():
#     for dataset in DATASETS:
#         _, TOPICS_NAME, DATASET_SHORT = ds_config(dataset)
#         qrels = get_qrels(TOPICS_NAME)

#         for model in MODELS:
#             results_root = f"results/{DATASET_SHORT}_{model}"
#             if not os.path.isdir(results_root):
#                 print(f"⚠️  Missing: {results_root}")
#                 continue

#             print(f"\n=== Analyzing {dataset} × {model} ===")
#             ordering_dirs = _paths_for_ordering(results_root)

#             records = []
#             for ordering in ORDERINGS:
#                 root_path = ordering_dirs.get(ordering)
#                 if not root_path or not os.path.isdir(root_path):
#                     print(f"⚠️  Missing path for ordering={ordering}: {root_path}")
#                     continue

#                 try:
#                     qids = sorted([d for d in os.listdir(root_path)
#                                    if os.path.isdir(os.path.join(root_path, d))])
#                 except Exception:
#                     continue

#                 for qid in tqdm(qids, desc=f"{dataset}_{model} {ordering}"):
#                     qdir = os.path.join(root_path, qid)
#                     qrel_dict = get_qrel_dict_robust(qrels, qid, warn_if_missing=True, tag="ana-qrels")

#                     for inp_idx in range(NUM_INPUTS):
#                         f0 = os.path.join(qdir, f"in{inp_idx:02d}_run0.json")
#                         f1 = os.path.join(qdir, f"in{inp_idx:02d}_run1.json")
#                         if not (os.path.exists(f0) and os.path.exists(f1)):
#                             continue

#                         j0 = _load_json(f0)
#                         j1 = _load_json(f1)
#                         if not j0 or not j1:
#                             continue

#                         # always read inputs from each log (both runs should share same inputs)
#                         in_ids = j0.get("input_docids", []) or j1.get("input_docids", [])
#                         if not in_ids:
#                             # last resort: skip silently
#                             continue

#                         out0 = j0.get("output_docids", []) or []
#                         out1 = j1.get("output_docids", []) or []

#                         # sanitize to guarantee: same set, length=K, subset of inputs, no dup
#                         out0 = _sanitize_output(out0, in_ids, K)
#                         out1 = _sanitize_output(out1, in_ids, K)

#                         # now safe to compute τ, RBO, nDCG
#                         tau  = kendall_tau_on_ids(out0, out1)
#                         rbo  = rbo_score(out0, out1, RBO_P)
#                         nd0  = ndcg_for_out(out0, qrel_dict, K)
#                         nd1  = ndcg_for_out(out1, qrel_dict, K)
#                         ndcg = 0.5 * (nd0 + nd1)

#                         records.append({
#                             "ordering": ordering,
#                             "tau": tau,
#                             "rbo": rbo,
#                             "ndcg": ndcg
#                         })

#             df = pd.DataFrame(records)
#             if df.empty:
#                 print(f"⚠️  No usable records for {dataset}×{model}.")
#                 continue

#             summary = (
#                 df.groupby("ordering")[["tau", "rbo", "ndcg"]]
#                   .mean()
#                   .reindex(ORDERINGS, fill_value=np.nan)
#                   .reset_index()
#             )

#             out_path = os.path.join(results_root, "results_order_summary.txt")
#             with open(out_path, "w") as out:
#                 out.write("Aggregated Kendall τ, RBO@0.79, nDCG@10 per ordering\n")
#                 out.write("=" * 60 + "\n")
#                 out.write(f"Dataset: {dataset}\nModel: {model}\n\n")
#                 out.write(f"{'Ordering':<12} {'τ':>8} {'RBO':>8} {'nDCG@10':>10}\n")
#                 out.write("-" * 42 + "\n")
#                 for _, row in summary.iterrows():
#                     out.write(
#                         f"{row['ordering']:<12} "
#                         f"{(row['tau'] if pd.notna(row['tau']) else float('nan')):>8.3f} "
#                         f"{(row['rbo'] if pd.notna(row['rbo']) else float('nan')):>8.3f} "
#                         f"{(row['ndcg'] if pd.notna(row['ndcg']) else float('nan')):>10.3f}\n"
#                     )
#             print(f"✅ Summary saved to {out_path}")



################################## try to match RQ@ -- still same as befor ###################################
###### dosent matter use this version or previous one that is commented ##################################


import os
import re
import json
from collections import OrderedDict

from scipy.stats import mannwhitneyu, ttest_ind
import numpy as np
import pandas as pd
from tqdm import tqdm
# assume RankingSimilarity, DATASETS, MODELS, ds_config, get_qrels,
# get_qrel_dict_robust, NUM_INPUTS, BASE_MODE, ORDERINGS, RBO_P,
# ndcg_for_out, K are defined elsewhere

# ---------------------------------------------------------------------
# Effect size & significance helpers
# ---------------------------------------------------------------------


def _safe_rel(qrel_dict, docid):
    """
    Get relevance as a float, robust to string/None/etc.
    """
    val = qrel_dict.get(docid, 0.0)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0

    
    
def _cliffs_delta(x, y):
    # Cliff's delta in O(n log n)
    x = np.asarray(x); y = np.asarray(y)
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    nx, ny = len(x_sorted), len(y_sorted)
    i = j = more = less = 0
    while i < nx and j < ny:
        if x_sorted[i] > y_sorted[j]:
            more += ny - j
            i += 1
        elif x_sorted[i] < y_sorted[j]:
            less += ny - j
            i += 1
        else:
            v = y_sorted[j]
            j2 = j
            while j2 < ny and y_sorted[j2] == v:
                j2 += 1
            # ties neither > nor <
            i += 1
            j = j2
    return (more - less) / (nx * ny)


def _concise_sig_line(df, metric_col, pretty_name):
    """
    best-vs-worst significance for a single metric column.
    """
    if metric_col not in df:
        return f"{pretty_name}: metric column '{metric_col}' not in dataframe\n"

    means = df.groupby("ordering")[metric_col].mean().sort_values()
    if len(means.index) < 2:
        return f"{pretty_name}: not enough orderings for best/worst test\n"

    worst, best = means.index[0], means.index[-1]
    x = df.loc[df["ordering"] == best, metric_col].dropna().values
    y = df.loc[df["ordering"] == worst, metric_col].dropna().values

    if len(x) < 2 or len(y) < 2:
        return f"{pretty_name}: best={best}, worst={worst} — insufficient data for test\n"

    u_stat, p_u = mannwhitneyu(x, y, alternative="two-sided")
    t_stat, p_t = ttest_ind(x, y, equal_var=False)
    delta = _cliffs_delta(x, y)

    significant = (p_u < 0.05) and (p_t < 0.05)
    verdict = "SIGNIFICANT" if significant else "not significant"
    return (f"{pretty_name}: best={best}, worst={worst} — {verdict} "
            f"(U p={p_u:.2g}, t p={p_t:.2g}, δ={delta:.2f})\n")


def _fmt_p(p):
    return f"{p:.4g}"


def _append_pairwise_top_vs_bottom(out_path, df, label):
    """
    Generic helper for rel_top vs rel_bottom on a given metric column `label`.
    """
    if label not in df:
        with open(out_path, "a") as out:
            out.write(f"{label}: column not in dataframe; skipping rel_top vs rel_bottom\n")
        return

    a = df[df["ordering"] == "rel_top"][label].dropna().to_numpy()
    b = df[df["ordering"] == "rel_bottom"][label].dropna().to_numpy()

    if len(a) < 2 or len(b) < 2:
        line = (f"{label}: insufficient data to test rel_top vs rel_bottom "
                f"(|A|={len(a)}, |B|={len(b)})\n")
        with open(out_path, "a") as out:
            out.write(line)
        return

    mean_a, mean_b = np.mean(a), np.mean(b)
    p_u = mannwhitneyu(a, b, alternative="two-sided").pvalue
    p_t = ttest_ind(a, b, equal_var=False).pvalue
    delta = _cliffs_delta(a, b)

    better = "rel_top" if mean_a > mean_b else "rel_bottom"
    significant = (p_u < 0.05 and p_t < 0.05)
    verdict = "SIGNIFICANT" if significant else "not significant"

    with open(out_path, "a") as out:
        out.write(
            f"{label}: rel_top (mean={mean_a:.3f}) vs rel_bottom (mean={mean_b:.3f}) "
            f"→ {better} better; {verdict} "
            f"[Mann–Whitney p={_fmt_p(p_u)}, Welch t p={_fmt_p(p_t)}, Cliff's δ={delta:.3f}]\n"
        )


def _append_pairwise_ordering(out_path, df, label, ord_a, ord_b):
    """
    Generic pairwise test between two specific orderings for a given metric.
    """
    if label not in df:
        with open(out_path, "a") as out:
            out.write(f"{label}: column not in dataframe; skipping {ord_a} vs {ord_b}\n")
        return

    a = df[df["ordering"] == ord_a][label].dropna().to_numpy()
    b = df[df["ordering"] == ord_b][label].dropna().to_numpy()

    if len(a) < 2 or len(b) < 2:
        line = (f"{label}: insufficient data to test {ord_a} vs {ord_b} "
                f"(|A|={len(a)}, |B|={len(b)})\n")
        with open(out_path, "a") as out:
            out.write(line)
        return

    mean_a, mean_b = np.mean(a), np.mean(b)
    p_u = mannwhitneyu(a, b, alternative="two-sided").pvalue
    p_t = ttest_ind(a, b, equal_var=False).pvalue
    delta = _cliffs_delta(a, b)

    better = ord_a if mean_a > mean_b else ord_b
    significant = (p_u < 0.05 and p_t < 0.05)
    verdict = "SIGNIFICANT" if significant else "not significant"

    with open(out_path, "a") as out:
        out.write(
            f"{label}: {ord_a} (mean={mean_a:.3f}) vs {ord_b} (mean={mean_b:.3f}) "
            f"→ {better} better; {verdict} "
            f"[Mann–Whitney p={_fmt_p(p_u)}, Welch t p={_fmt_p(p_t)}, Cliff's δ={delta:.3f}]\n"
        )

# ---------------------------------------------------------------------
# Extra ranking metrics: DCG, sDCG (binary), RR
# ---------------------------------------------------------------------


def _dcg_for_out(docids, qrel_dict, k):
    """
    DCG with *linear* gain: rel / log2(rank+1),
    matching your version-2 behaviour.
    """
    dcg = 0.0
    for rank, docid in enumerate(docids[:k], start=1):
        rel = _safe_rel(qrel_dict, docid)
        # no need to skip non-positive rel; they add 0
        dcg += rel / np.log2(rank + 1.0)
    return float(dcg)



def _sdcg_for_out(docids, qrel_dict, k, max_rel=2.0):
    """
    Graded, *scaled* DCG:
      - Numerator: DCG@k with graded relevance (linear gain, rel/log2(rank+1)).
      - Denominator: DCG@k if ALL docs up to k had relevance = max_rel.
        (So ideal_labels = [max_rel, max_rel, ..., max_rel])
      - Output is in [0, 1].
    """
    n = min(k, len(docids))
    if n == 0:
        return 0.0

    # graded DCG of current list
    labels = [_safe_rel(qrel_dict, d) for d in docids[:n]]
    num_dcg = _dcg_from_labels(labels, n)

    # ideal DCG if all were max_rel
    ideal_labels = [max_rel] * n
    denom_dcg = _dcg_from_labels(ideal_labels, n)

    return float(num_dcg / denom_dcg) if denom_dcg > 0.0 else 0.0



def _rr_for_out(docids, qrel_dict, k):
    """
    Reciprocal Rank: 1/rank of first relevant (rel>0), else 0.
    """
    for rank, docid in enumerate(docids[:k], start=1):
        rel = _safe_rel(qrel_dict, docid)
        if rel > 0.0:
            return 1.0 / float(rank)
    return 0.0
        
# ---------------------------------------------------------------------
# RQ2 helper (unchanged in behaviour)
# ---------------------------------------------------------------------

def _read_rq2_half_n10_metrics(results_root):
    """
    Parse results_summary_shared_ids.txt and extract:
      half_relevant N10  -> τ@10, RBO@10, nDCG@k
    Returns (tau10, rbo10, ndcg) as floats, or None if not found.
    """
    path = os.path.join(results_root, "results_summary_shared_ids.txt")
    if not os.path.exists(path):
        return None

    pat = re.compile(
        r'^half_relevant\s+N10\s+'
        r'([0-9.]+)\s+([0-9.]+)\s+'   # τ@5, RBO@5  (ignored)
        r'([0-9.]+)\s+([0-9.]+)\s+'   # τ@10, RBO@10 <-- we want these
        r'([0-9.]+)\s+([0-9.]+)\s+'   # τ@20, RBO@20 (ignored)
        r'([0-9.]+)\s*$'              # nDCG@k       <-- we want this
    )

    with open(path, "r") as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                tau10 = float(m.group(3))
                rbo10 = float(m.group(4))
                ndcg  = float(m.group(7))
                return (tau10, rbo10, ndcg)
    return None

# ---------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------

def compute_order_metrics():
    from collections import OrderedDict as _OD

    def _load_json(p):
        try:
            with open(p, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def _dedup_preserve(seq):
        return list(_OD.fromkeys(seq or []))

    def _sanitize_to_k(out_ids, allowed_ids_set, k):
        # keep only unique items that are in inputs, then pad from inputs to reach k
        seen, filtered = set(), []
        for d in out_ids or []:
            if d in allowed_ids_set and d not in seen:
                filtered.append(d); seen.add(d)
            if len(filtered) == k:
                break
        if len(filtered) < k:
            for d in allowed_ids_set:
                if d not in seen:
                    filtered.append(d); seen.add(d)
                if len(filtered) == k:
                    break
        return filtered

    def _kendall_tau_on_shared(out0, out1, shared_ids):
        # order out0 as baseline, project out1 onto it
        a = [d for d in out0 if d in shared_ids]
        b = [d for d in out1 if d in shared_ids]
        if len(a) < 2 or len(b) < 2:
            return float("nan")
        pos = {d: i for i, d in enumerate(a)}
        seq_a = [pos[d] for d in b if d in pos]
        if len(seq_a) < 2:
            return float("nan")
        from scipy.stats import kendalltau
        tau_val, _ = kendalltau(seq_a, list(range(len(seq_a))))
        return float(tau_val)

    def _rbo_on_shared(out0, out1, shared_ids, p):
        # RBO implementation requires unique lists; restrict to shared first
        a = [d for d in out0 if d in shared_ids]
        b = [d for d in out1 if d in shared_ids]
        a = list(_OD.fromkeys(a))
        b = list(_OD.fromkeys(b))
        if not a or not b:
            return float("nan")
        return RankingSimilarity(a, b).rbo(p=p)

    # ---- main loop ----
    for dataset in DATASETS:
        _, TOPICS_NAME, DATASET_SHORT = ds_config(dataset)
        qrels = get_qrels(TOPICS_NAME)

        for model in MODELS:
            results_root = f"results/{DATASET_SHORT}_{model}"
            if not os.path.isdir(results_root):
                print(f"⚠️  Skipping {dataset}×{model}: folder not found.")
                continue

            print(f"\n=== Analyzing {dataset} × {model} ===")
            records = []

            shuffle_dir   = os.path.join(results_root, "N10", BASE_MODE)   # half_relevant @ N10
            shuffle_dir20 = os.path.join(results_root, "N20", BASE_MODE)   # half_relevant @ N20 (for shared10)
            order_dirs = {
                "rel_top":    os.path.join(results_root, "order_exp", "rel_top"),
                "rel_bottom": os.path.join(results_root, "order_exp", "rel_bottom"),
                "rel_middle": os.path.join(results_root, "order_exp", "rel_middle"),
                "rel_ends":   os.path.join(results_root, "order_exp", "rel_ends"),
            }

            # ---- SHUFFLE: match RQ2 (use shared between N10 and N20) ----
            if os.path.isdir(shuffle_dir):
                qids = sorted([q for q in os.listdir(shuffle_dir)
                               if os.path.isdir(os.path.join(shuffle_dir, q))])
                for qid in tqdm(qids, desc=f"{dataset}_{model} shuffle"):
                    qrel_dict = get_qrel_dict_robust(qrels, qid)

                    for i in range(NUM_INPUTS):
                        f0 = os.path.join(shuffle_dir, qid, f"in{i:02d}_run0.json")
                        f1 = os.path.join(shuffle_dir, qid, f"in{i:02d}_run1.json")
                        jin0, jin1 = _load_json(f0), _load_json(f1)
                        if not (jin0 and jin1):
                            continue

                        # original input ranking for metrics (N10)
                        in_list = (jin0.get("input_docids", []) or [])[:K]
                        in10 = set(in_list)

                        # inputs for shared computation (N10 ∩ N20)
                        if os.path.isdir(os.path.join(shuffle_dir20, qid)):
                            f20 = os.path.join(shuffle_dir20, qid, f"in{i:02d}_run0.json")
                            j20 = _load_json(f20)
                            in20 = set((j20 or {}).get("input_docids", [])[:20])
                        else:
                            in20 = set()
                        shared10 = (in10 & in20) if in20 else in10

                        allowed = in10  # for shuffle at k=10, only N10 inputs are allowed
                        out0 = _sanitize_to_k(_dedup_preserve(jin0.get("output_docids", [])), allowed, K)
                        out1 = _sanitize_to_k(_dedup_preserve(jin1.get("output_docids", [])), allowed, K)

                        # τ / RBO on shared10
                        tau = _kendall_tau_on_shared(out0, out1, shared10)
                        rbo = _rbo_on_shared(out0, out1, shared10, RBO_P)

                        # ranking quality metrics: input vs outputs (mean over two runs)
                        nd_in = ndcg_for_out(in_list, qrel_dict, K)
                        nd0   = ndcg_for_out(out0,    qrel_dict, K)
                        nd1   = ndcg_for_out(out1,    qrel_dict, K)
                        ndcg  = float(np.mean([nd0, nd1]))
                        d_nd  = ndcg - nd_in

                        dcg_in = _dcg_for_out(in_list, qrel_dict, K)
                        dcg0   = _dcg_for_out(out0,    qrel_dict, K)
                        dcg1   = _dcg_for_out(out1,    qrel_dict, K)
                        dcg    = float(np.mean([dcg0, dcg1]))
                        d_dcg  = dcg - dcg_in

                        sdcg_in = _sdcg_for_out(in_list, qrel_dict, K)
                        sdcg0   = _sdcg_for_out(out0,    qrel_dict, K)
                        sdcg1   = _sdcg_for_out(out1,    qrel_dict, K)
                        sdcg    = float(np.mean([sdcg0, sdcg1]))
                        d_sdcg  = sdcg - sdcg_in

                        rr_in = _rr_for_out(in_list, qrel_dict, K)
                        rr0   = _rr_for_out(out0,    qrel_dict, K)
                        rr1   = _rr_for_out(out1,    qrel_dict, K)
                        rr    = float(np.mean([rr0, rr1]))
                        d_rr  = rr - rr_in

                        if not np.isnan(tau):
                            records.append({
                                "ordering":    "shuffle",
                                "tau":         tau,
                                "rbo":         rbo,
                                "ndcg":        ndcg,
                                "delta_ndcg":  d_nd,
                                "dcg":         dcg,
                                "delta_dcg":   d_dcg,
                                "sdcg":        sdcg,
                                "delta_sdcg":  d_sdcg,
                                "rr":          rr,
                                "delta_rr":    d_rr,
                            })

            # ---- ORDERED VARIANTS: evaluate on their own inputs (no N20) ----
            for ordering, root_path in order_dirs.items():
                if not os.path.isdir(root_path):
                    continue
                qids = sorted([q for q in os.listdir(root_path)
                               if os.path.isdir(os.path.join(root_path, q))])
                for qid in tqdm(qids, desc=f"{dataset}_{model} {ordering}"):
                    qrel_dict = get_qrel_dict_robust(qrels, qid)

                    for i in range(NUM_INPUTS):
                        f0 = os.path.join(root_path, qid, f"in{i:02d}_run0.json")
                        f1 = os.path.join(root_path, qid, f"in{i:02d}_run1.json")
                        j0, j1 = _load_json(f0), _load_json(f1)
                        if not (j0 and j1):
                            continue

                        in_list = (j0.get("input_docids", []) or [])[:K]
                        in_ids  = set(in_list)
                        out0 = _sanitize_to_k(_dedup_preserve(j0.get("output_docids", [])), in_ids, K)
                        out1 = _sanitize_to_k(_dedup_preserve(j1.get("output_docids", [])), in_ids, K)

                        shared = in_ids  # for orderings, "shared" is all their own inputs
                        tau = _kendall_tau_on_shared(out0, out1, shared)
                        rbo = _rbo_on_shared(out0, out1, shared, RBO_P)

                        nd_in = ndcg_for_out(in_list, qrel_dict, K)
                        nd0   = ndcg_for_out(out0,    qrel_dict, K)
                        nd1   = ndcg_for_out(out1,    qrel_dict, K)
                        ndcg  = float(np.mean([nd0, nd1]))
                        d_nd  = ndcg - nd_in

                        dcg_in = _dcg_for_out(in_list, qrel_dict, K)
                        dcg0   = _dcg_for_out(out0,    qrel_dict, K)
                        dcg1   = _dcg_for_out(out1,    qrel_dict, K)
                        dcg    = float(np.mean([dcg0, dcg1]))
                        d_dcg  = dcg - dcg_in

                        sdcg_in = _sdcg_for_out(in_list, qrel_dict, K)
                        sdcg0   = _sdcg_for_out(out0,    qrel_dict, K)
                        sdcg1   = _sdcg_for_out(out1,    qrel_dict, K)
                        sdcg    = float(np.mean([sdcg0, sdcg1]))
                        d_sdcg  = sdcg - sdcg_in

                        rr_in = _rr_for_out(in_list, qrel_dict, K)
                        rr0   = _rr_for_out(out0,    qrel_dict, K)
                        rr1   = _rr_for_out(out1,    qrel_dict, K)
                        rr    = float(np.mean([rr0, rr1]))
                        d_rr  = rr - rr_in

                        if not np.isnan(tau):
                            records.append({
                                "ordering":    ordering,
                                "tau":         tau,
                                "rbo":         rbo,
                                "ndcg":        ndcg,
                                "delta_ndcg":  d_nd,
                                "dcg":         dcg,
                                "delta_dcg":   d_dcg,
                                "sdcg":        sdcg,
                                "delta_sdcg":  d_sdcg,
                                "rr":          rr,
                                "delta_rr":    d_rr,
                            })

            # ---- Aggregate & write summary (still only τ/RBO/nDCG table) ----
            df = pd.DataFrame(records)
            if df.empty:
                print(f"⚠️  No records for {dataset}×{model}.")
                continue

            summary = (
                df.groupby("ordering")[["tau", "rbo", "ndcg"]]
                .mean()
                .reindex(["shuffle"] + ORDERINGS, fill_value=np.nan)
                .reset_index()
            )

            out_path = os.path.join(results_root, "results_order_summary.txt")
            with open(out_path, "w") as out:
                out.write("Aggregated τ (shared), RBO@0.79 (shared), nDCG@10 (full) per ordering\n")
                out.write("=" * 70 + "\n")
                out.write(f"Dataset: {dataset}\nModel: {model}\n\n")
                out.write(f"{'Ordering':<12} {'τ@shared':>10} {'RBO@0.79':>10} {'nDCG@10':>10}\n")
                out.write("-" * 48 + "\n")
                for _, row in summary.iterrows():
                    if pd.isna(row["tau"]):
                        continue
                    out.write(
                        f"{row['ordering']:<12} "
                        f"{row['tau']:.3f}     {row['rbo']:.3f}     {row['ndcg']:.3f}\n"
                    )

            # append RQ2 shuffle line if available
            rq2_vals = _read_rq2_half_n10_metrics(results_root)
            if rq2_vals is not None:
                tau10_rq2, rbo10_rq2, ndcg_rq2 = rq2_vals
                with open(out_path, "a") as out:
                    out.write(
                        f"{'shuffle (RQ2 N10)':<12} "
                        f"{tau10_rq2:.3f}   {rbo10_rq2:.3f}   {ndcg_rq2:.3f}\n"
                    )

            print(f"✅ Summary saved to {out_path}")

            # --- Significance sections (appended; file not overwritten) ---
            with open(out_path, "a") as out:
                out.write("\nSignificance (best vs worst per metric):\n")
                out.write(_concise_sig_line(df, "tau",          "tau"))
                out.write(_concise_sig_line(df, "rbo",          "rbo"))
                out.write(_concise_sig_line(df, "ndcg",         "nDCG"))
                out.write(_concise_sig_line(df, "delta_ndcg",   "ΔnDCG"))
                out.write(_concise_sig_line(df, "dcg",          "DCG"))
                out.write(_concise_sig_line(df, "delta_dcg",    "ΔDCG"))
                out.write(_concise_sig_line(df, "sdcg",         "sDCG"))
                out.write(_concise_sig_line(df, "delta_sdcg",   "ΔsDCG"))
                out.write(_concise_sig_line(df, "rr",           "RR"))
                out.write(_concise_sig_line(df, "delta_rr",     "ΔRR"))

                out.write("\nPairwise significance: rel_top vs rel_bottom\n")
                out.write("(Two-sided Mann–Whitney U, Welch t; Cliff's δ reported)\n")

            # rel_top vs rel_bottom for all metrics
            for label in ["tau", "rbo", "ndcg",
                          "delta_ndcg", "dcg", "delta_dcg",
                          "sdcg", "delta_sdcg", "rr", "delta_rr"]:
                _append_pairwise_top_vs_bottom(out_path, df, label)

            # selected additional pairwise comparisons
            with open(out_path, "a") as out:
                out.write("\nPairwise significance: selected ordering comparisons\n")
                out.write("(Two-sided Mann–Whitney U, Welch t; Cliff's δ reported)\n")

            pair_tests = [
                # rel_top vs others (rel_top vs rel_bottom already done separately)
                ("rel_top",    "rel_middle"),
                ("rel_top",    "rel_ends"),
                ("rel_top",    "shuffle"),
                ("rel_top",   "rel_bottom"),

                # rel_middle vs rel_bottom, rel_ends, shuffle
                ("rel_middle", "rel_bottom"),
                ("rel_middle", "rel_ends"),
                ("rel_middle", "shuffle"),

                # rel_ends vs rel_bottom, rel_middle, shuffle
                ("rel_ends",   "rel_bottom"),
                ("rel_ends",   "rel_middle"),
                ("rel_ends",   "shuffle"),

                # shuffle vs rel_bottom, rel_middle, rel_ends
                ("shuffle",    "rel_bottom"),
                ("shuffle",    "rel_middle"),
                ("shuffle",    "rel_ends"),
                ("shuffle",    "rel_top"),
            ]

            metrics_for_pairs = ["tau", "rbo", "ndcg",
                                 "delta_ndcg", "dcg", "delta_dcg",
                                 "sdcg", "delta_sdcg", "rr", "delta_rr"]

            for (ord_a, ord_b) in pair_tests:
                for label in metrics_for_pairs:
                    _append_pairwise_ordering(out_path, df, label, ord_a, ord_b)




# ───────────────────────────────
# Run whichever you need
# ───────────────────────────────
# # Uncomment one of these:
# generate_order_logs()
# compute_order_metrics()




######################################################################################################################
#################### like previous but this version has ERR and SDCG with graded  relevance and all rel =2 ###########
########################################################################################################################

# ---------------------------------------------------------------------
# Extra ranking metrics: DCG (graded), sDCG (graded, scaled), RR, ERR, IDCG
# ---------------------------------------------------------------------

def _safe_rel(qrel_dict, docid):
    """
    Get relevance as a float, robust to string/None/etc.
    """
    val = qrel_dict.get(docid, 0.0)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _dcg_from_labels(labels, k):
    """DCG with linear gain: rel / log2(rank+1)."""
    dcg = 0.0
    for rank, rel in enumerate(labels[:k], start=1):
        dcg += rel / np.log2(rank + 1.0)
    return float(dcg)


def _dcg_for_out(docids, qrel_dict, k):
    """
    DCG with *linear* gain: rel / log2(rank+1),
    matching your version-2 behaviour.
    """
    labels = [_safe_rel(qrel_dict, d) for d in docids[:k]]
    return _dcg_from_labels(labels, k)


def _idcg_for_out(docids, qrel_dict, k):
    """
    Ideal DCG@k (IDCG) with linear gain:
    - Uses the graded labels of the given docids
    - Sorts them descending and computes DCG.
    """
    labels = [_safe_rel(qrel_dict, d) for d in docids[:k]]
    if not labels:
        return 0.0
    ideal_labels = sorted(labels, reverse=True)
    return _dcg_from_labels(ideal_labels, k)


def _sdcg_for_out(docids, qrel_dict, k, max_rel=2.0):
    """
    Graded, *scaled* DCG:
      - Numerator: DCG@k with graded relevance (linear gain, rel/log2(rank+1)).
      - Denominator: DCG@k if ALL docs up to k had relevance = max_rel.
        (So ideal_labels = [max_rel, max_rel, ..., max_rel])
      - Output is in [0, 1].
    """
    n = min(k, len(docids))
    if n == 0:
        return 0.0

    # graded DCG of current list
    labels = [_safe_rel(qrel_dict, d) for d in docids[:n]]
    num_dcg = _dcg_from_labels(labels, n)

    # ideal DCG if all were max_rel
    ideal_labels = [max_rel] * n
    denom_dcg = _dcg_from_labels(ideal_labels, n)

    return float(num_dcg / denom_dcg) if denom_dcg > 0.0 else 0.0


def _rr_for_out(docids, qrel_dict, k):
    """
    Reciprocal Rank: 1/rank of first relevant (rel>0), else 0.
    """
    for rank, docid in enumerate(docids[:k], start=1):
        rel = _safe_rel(qrel_dict, docid)
        if rel > 0.0:
            return 1.0 / float(rank)
    return 0.0


def _err_for_out(docids, qrel_dict, k, max_rel=2.0):
    """
    Expected Reciprocal Rank (ERR@k) with graded relevance.

    Standard formulation:
      - R(rel) = (2^rel - 1) / (2^max_rel)
      - ERR = sum_{r=1..k} P(continue up to r-1) * R(rel_r) / r
    """
    if not docids:
        return 0.0

    p_continue = 1.0
    err = 0.0

    for rank, docid in enumerate(docids[:k], start=1):
        rel = _safe_rel(qrel_dict, docid)
        if rel < 0.0:
            rel = 0.0
        if rel > max_rel:
            rel = max_rel

        R = (2.0**rel - 1.0) / (2.0**max_rel)
        err += p_continue * (R / float(rank))
        p_continue *= (1.0 - R)

        if p_continue <= 1e-8:
            break

    return float(err)


def compute_order_metrics():
    from collections import OrderedDict as _OD

    def _load_json(p):
        try:
            with open(p, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def _dedup_preserve(seq):
        return list(_OD.fromkeys(seq or []))

    def _sanitize_to_k(out_ids, allowed_ids_set, k):
        # keep only unique items that are in inputs, then pad from inputs to reach k
        seen, filtered = set(), []
        for d in out_ids or []:
            if d in allowed_ids_set and d not in seen:
                filtered.append(d)
                seen.add(d)
            if len(filtered) == k:
                break
        if len(filtered) < k:
            for d in allowed_ids_set:
                if d not in seen:
                    filtered.append(d)
                    seen.add(d)
                if len(filtered) == k:
                    break
        return filtered

    def _kendall_tau_on_shared(out0, out1, shared_ids):
        # order out0 as baseline, project out1 onto it
        a = [d for d in out0 if d in shared_ids]
        b = [d for d in out1 if d in shared_ids]
        if len(a) < 2 or len(b) < 2:
            return float("nan")
        pos = {d: i for i, d in enumerate(a)}
        seq_a = [pos[d] for d in b if d in pos]
        if len(seq_a) < 2:
            return float("nan")
        from scipy.stats import kendalltau
        tau_val, _ = kendalltau(seq_a, list(range(len(seq_a))))
        return float(tau_val)

    def _rbo_on_shared(out0, out1, shared_ids, p):
        # RBO implementation requires unique lists; restrict to shared first
        a = [d for d in out0 if d in shared_ids]
        b = [d for d in out1 if d in shared_ids]
        a = list(_OD.fromkeys(a))
        b = list(_OD.fromkeys(b))
        if not a or not b:
            return float("nan")
        return RankingSimilarity(a, b).rbo(p=p)

    # ---- main loop ----
    for dataset in DATASETS:
        _, TOPICS_NAME, DATASET_SHORT = ds_config(dataset)
        qrels = get_qrels(TOPICS_NAME)

        for model in MODELS:
            results_root = f"results/{DATASET_SHORT}_{model}"
            if not os.path.isdir(results_root):
                print(f"⚠️  Skipping {dataset}×{model}: folder not found.")
                continue

            print(f"\n=== Analyzing {dataset} × {model} ===")
            records = []

            shuffle_dir = os.path.join(results_root, "N10", BASE_MODE)     # half_relevant @ N10
            shuffle_dir20 = os.path.join(results_root, "N20", BASE_MODE)   # half_relevant @ N20 (for shared10)
            order_dirs = {
                "rel_top":    os.path.join(results_root, "order_exp", "rel_top"),
                "rel_bottom": os.path.join(results_root, "order_exp", "rel_bottom"),
                "rel_middle": os.path.join(results_root, "order_exp", "rel_middle"),
                "rel_ends":   os.path.join(results_root, "order_exp", "rel_ends"),
            }

            # ---- SHUFFLE: match RQ2 (use shared between N10 and N20) ----
            if os.path.isdir(shuffle_dir):
                qids = sorted([q for q in os.listdir(shuffle_dir)
                               if os.path.isdir(os.path.join(shuffle_dir, q))])
                for qid in tqdm(qids, desc=f"{dataset}_{model} shuffle"):
                    qrel_dict = get_qrel_dict_robust(qrels, qid)

                    for i in range(NUM_INPUTS):
                        f0 = os.path.join(shuffle_dir, qid, f"in{i:02d}_run0.json")
                        f1 = os.path.join(shuffle_dir, qid, f"in{i:02d}_run1.json")
                        jin0, jin1 = _load_json(f0), _load_json(f1)
                        if not (jin0 and jin1):
                            continue

                        # original input ranking for metrics (N10)
                        in_list = (jin0.get("input_docids", []) or [])[:K]
                        in10 = set(in_list)

                        # inputs for shared computation (N10 ∩ N20)
                        if os.path.isdir(os.path.join(shuffle_dir20, qid)):
                            f20 = os.path.join(shuffle_dir20, qid, f"in{i:02d}_run0.json")
                            j20 = _load_json(f20)
                            in20 = set((j20 or {}).get("input_docids", [])[:20])
                        else:
                            in20 = set()
                        shared10 = (in10 & in20) if in20 else in10

                        allowed = in10  # for shuffle at k=10, only N10 inputs are allowed
                        out0 = _sanitize_to_k(_dedup_preserve(jin0.get("output_docids", [])), allowed, K)
                        out1 = _sanitize_to_k(_dedup_preserve(jin1.get("output_docids", [])), allowed, K)

                        # τ / RBO on shared10
                        tau = _kendall_tau_on_shared(out0, out1, shared10)
                        rbo = _rbo_on_shared(out0, out1, shared10, RBO_P)

                        # ranking quality metrics: input vs outputs (mean over two runs)

                        # nDCG (linear-gain variant, defined elsewhere)
                        nd_in = ndcg_for_out(in_list, qrel_dict, K)
                        nd0   = ndcg_for_out(out0,    qrel_dict, K)
                        nd1   = ndcg_for_out(out1,    qrel_dict, K)
                        ndcg  = float(np.mean([nd0, nd1]))
                        d_nd  = ndcg - nd_in

                        # DCG (graded, linear gain)
                        dcg_in = _dcg_for_out(in_list, qrel_dict, K)
                        dcg0   = _dcg_for_out(out0,    qrel_dict, K)
                        dcg1   = _dcg_for_out(out1,    qrel_dict, K)
                        dcg    = float(np.mean([dcg0, dcg1]))
                        d_dcg  = dcg - dcg_in

                        # sDCG (graded, scaled with max_rel=2)
                        sdcg_in = _sdcg_for_out(in_list, qrel_dict, K, max_rel=2.0)
                        sdcg0   = _sdcg_for_out(out0,    qrel_dict, K, max_rel=2.0)
                        sdcg1   = _sdcg_for_out(out1,    qrel_dict, K, max_rel=2.0)
                        sdcg    = float(np.mean([sdcg0, sdcg1]))
                        d_sdcg  = sdcg - sdcg_in

                        # RR
                        rr_in = _rr_for_out(in_list, qrel_dict, K)
                        rr0   = _rr_for_out(out0,    qrel_dict, K)
                        rr1   = _rr_for_out(out1,    qrel_dict, K)
                        rr    = float(np.mean([rr0, rr1]))
                        d_rr  = rr - rr_in

                        # ERR (graded)
                        err_in = _err_for_out(in_list, qrel_dict, K, max_rel=2.0)
                        err0   = _err_for_out(out0,    qrel_dict, K, max_rel=2.0)
                        err1   = _err_for_out(out1,    qrel_dict, K, max_rel=2.0)
                        err    = float(np.mean([err0, err1]))
                        d_err  = err - err_in

                        # IDCG (ideal DCG of the input list)
                        idcg_in = _idcg_for_out(in_list, qrel_dict, K)

                        if not np.isnan(tau):
                            records.append({
                                "ordering":    "shuffle",
                                "tau":         tau,
                                "rbo":         rbo,
                                "ndcg":        ndcg,
                                "delta_ndcg":  d_nd,
                                "dcg":         dcg,
                                "delta_dcg":   d_dcg,
                                "sdcg":        sdcg,
                                "delta_sdcg":  d_sdcg,
                                "rr":          rr,
                                "delta_rr":    d_rr,
                                "err":         err,
                                "delta_err":   d_err,
                                "idcg":        idcg_in,
                            })

            # ---- ORDERED VARIANTS: evaluate on their own inputs (no N20) ----
            for ordering, root_path in order_dirs.items():
                if not os.path.isdir(root_path):
                    continue
                qids = sorted([q for q in os.listdir(root_path)
                               if os.path.isdir(os.path.join(root_path, q))])
                for qid in tqdm(qids, desc=f"{dataset}_{model} {ordering}"):
                    qrel_dict = get_qrel_dict_robust(qrels, qid)

                    for i in range(NUM_INPUTS):
                        f0 = os.path.join(root_path, qid, f"in{i:02d}_run0.json")
                        f1 = os.path.join(root_path, qid, f"in{i:02d}_run1.json")
                        j0, j1 = _load_json(f0), _load_json(f1)
                        if not (j0 and j1):
                            continue

                        in_list = (j0.get("input_docids", []) or [])[:K]
                        in_ids  = set(in_list)
                        out0 = _sanitize_to_k(_dedup_preserve(j0.get("output_docids", [])), in_ids, K)
                        out1 = _sanitize_to_k(_dedup_preserve(j1.get("output_docids", [])), in_ids, K)

                        shared = in_ids  # for orderings, "shared" is all their own inputs
                        tau = _kendall_tau_on_shared(out0, out1, shared)
                        rbo = _rbo_on_shared(out0, out1, shared, RBO_P)

                        # nDCG
                        nd_in = ndcg_for_out(in_list, qrel_dict, K)
                        nd0   = ndcg_for_out(out0, qrel_dict, K)
                        nd1   = ndcg_for_out(out1, qrel_dict, K)
                        ndcg  = float(np.mean([nd0, nd1]))
                        d_nd  = ndcg - nd_in

                        # DCG
                        dcg_in = _dcg_for_out(in_list, qrel_dict, K)
                        dcg0   = _dcg_for_out(out0,    qrel_dict, K)
                        dcg1   = _dcg_for_out(out1,    qrel_dict, K)
                        dcg    = float(np.mean([dcg0, dcg1]))
                        d_dcg  = dcg - dcg_in

                        # sDCG (graded, scaled)
                        sdcg_in = _sdcg_for_out(in_list, qrel_dict, K, max_rel=2.0)
                        sdcg0   = _sdcg_for_out(out0,    qrel_dict, K, max_rel=2.0)
                        sdcg1   = _sdcg_for_out(out1,    qrel_dict, K, max_rel=2.0)
                        sdcg    = float(np.mean([sdcg0, sdcg1]))
                        d_sdcg  = sdcg - sdcg_in

                        # RR
                        rr_in = _rr_for_out(in_list, qrel_dict, K)
                        rr0   = _rr_for_out(out0,    qrel_dict, K)
                        rr1   = _rr_for_out(out1,    qrel_dict, K)
                        rr    = float(np.mean([rr0, rr1]))
                        d_rr  = rr - rr_in

                        # ERR
                        err_in = _err_for_out(in_list, qrel_dict, K, max_rel=2.0)
                        err0   = _err_for_out(out0,    qrel_dict, K, max_rel=2.0)
                        err1   = _err_for_out(out1,    qrel_dict, K, max_rel=2.0)
                        err    = float(np.mean([err0, err1]))
                        d_err  = err - err_in

                        # IDCG (for the input list)
                        idcg_in = _idcg_for_out(in_list, qrel_dict, K)

                        if not np.isnan(tau):
                            records.append({
                                "ordering":    ordering,
                                "tau":         tau,
                                "rbo":         rbo,
                                "ndcg":        ndcg,
                                "delta_ndcg":  d_nd,
                                "dcg":         dcg,
                                "delta_dcg":   d_dcg,
                                "sdcg":        sdcg,
                                "delta_sdcg":  d_sdcg,
                                "rr":          rr,
                                "delta_rr":    d_rr,
                                "err":         err,
                                "delta_err":   d_err,
                                "idcg":        idcg_in,
                            })

            # ---- Aggregate & write summary (τ/RBO/nDCG table with 6 decimals) ----
            df = pd.DataFrame(records)
            if df.empty:
                print(f"⚠️  No records for {dataset}×{model}.")
                continue

            summary = (
                df.groupby("ordering")[["tau", "rbo", "ndcg"]]
                .mean()
                .reindex(["shuffle"] + ORDERINGS, fill_value=np.nan)
                .reset_index()
            )

            out_path = os.path.join(results_root, "results_order_summary.txt")
            with open(out_path, "w") as out:
                out.write("Aggregated τ (shared), RBO@0.79 (shared), nDCG@10 (full) per ordering\n")
                out.write("=" * 70 + "\n")
                out.write(f"Dataset: {dataset}\nModel: {model}\n\n")
                out.write(f"{'Ordering':<12} {'τ@shared':>12} {'RBO@0.79':>12} {'nDCG@10':>12}\n")
                out.write("-" * 60 + "\n")
                for _, row in summary.iterrows():
                    if pd.isna(row["tau"]):
                        continue
                    out.write(
                        f"{row['ordering']:<12} "
                        f"{row['tau']:.6f}     {row['rbo']:.6f}     {row['ndcg']:.6f}\n"
                    )

            # append RQ2 shuffle line if available (also 6 decimals)
            rq2_vals = _read_rq2_half_n10_metrics(results_root)
            if rq2_vals is not None:
                tau10_rq2, rbo10_rq2, ndcg_rq2 = rq2_vals
                with open(out_path, "a") as out:
                    out.write(
                        f"{'shuffle (RQ2 N10)':<12} "
                        f"{tau10_rq2:.6f}   {rbo10_rq2:.6f}   {ndcg_rq2:.6f}\n"
                    )

            print(f"✅ Summary saved to {out_path}")

            # --- Significance sections (best vs worst, all metrics) ---
            with open(out_path, "a") as out:
                out.write("\nSignificance (best vs worst per metric):\n")
                out.write(_concise_sig_line(df, "tau",         "tau"))
                out.write(_concise_sig_line(df, "rbo",         "rbo"))
                out.write(_concise_sig_line(df, "ndcg",        "nDCG"))
                out.write(_concise_sig_line(df, "delta_ndcg",  "ΔnDCG"))
                out.write(_concise_sig_line(df, "dcg",         "DCG"))
                out.write(_concise_sig_line(df, "delta_dcg",   "ΔDCG"))
                out.write(_concise_sig_line(df, "sdcg",        "sDCG"))
                out.write(_concise_sig_line(df, "delta_sdcg",  "ΔsDCG"))
                out.write(_concise_sig_line(df, "rr",          "RR"))
                out.write(_concise_sig_line(df, "delta_rr",    "ΔRR"))
                out.write(_concise_sig_line(df, "err",         "ERR"))
                out.write(_concise_sig_line(df, "delta_err",   "ΔERR"))
                out.write(_concise_sig_line(df, "idcg",        "IDCG"))

                out.write("\nPairwise significance: rel_top vs rel_bottom\n")
                out.write("(Two-sided Mann–Whitney U, Welch t; Cliff's δ reported)\n")

            # rel_top vs rel_bottom for all metrics
            for label in [
                "tau", "rbo", "ndcg",
                "delta_ndcg", "dcg", "delta_dcg",
                "sdcg", "delta_sdcg",
                "rr", "delta_rr",
                "err", "delta_err",
                "idcg",
            ]:
                _append_pairwise_top_vs_bottom(out_path, df, label)

            # selected additional pairwise comparisons
            with open(out_path, "a") as out:
                out.write("\nPairwise significance: selected ordering comparisons\n")
                out.write("(Two-sided Mann–Whitney U, Welch t; Cliff's δ reported)\n")

            pair_tests = [
                # rel_top vs others (rel_top vs rel_bottom already done separately)
                ("rel_top",    "rel_middle"),
                ("rel_top",    "rel_ends"),
                ("rel_top",    "shuffle"),
                ("rel_top",    "rel_bottom"),

                # rel_middle vs rel_bottom, rel_ends, shuffle
                ("rel_middle", "rel_bottom"),
                ("rel_middle", "rel_ends"),
                ("rel_middle", "shuffle"),

                # rel_ends vs rel_bottom, rel_middle, shuffle
                ("rel_ends",   "rel_bottom"),
                ("rel_ends",   "rel_middle"),
                ("rel_ends",   "shuffle"),

                # shuffle vs rel_bottom, rel_middle, rel_ends, rel_top
                ("shuffle",    "rel_bottom"),
                ("shuffle",    "rel_middle"),
                ("shuffle",    "rel_ends"),
                ("shuffle",    "rel_top"),
            ]

            metrics_for_pairs = [
                "tau", "rbo", "ndcg",
                "delta_ndcg", "dcg", "delta_dcg",
                "sdcg", "delta_sdcg",
                "rr", "delta_rr",
                "err", "delta_err",
                "idcg",
            ]

            for (ord_a, ord_b) in pair_tests:
                for label in metrics_for_pairs:
                    _append_pairwise_ordering(out_path, df, label, ord_a, ord_b)


# generate_order_logs()
compute_order_metrics()


#############################################################################################################################
############################# Rbo and Tau between input and output ##########################################################
##############################################################################################################################


def compute_order_metrics_input_vs_output():
    """
    Like compute_order_metrics, but:
      - τ and RBO are computed BETWEEN INPUT and each output run (run0, run1),
        i.e., tau(input, out0) and tau(input, out1), then averaged per input.
      - For each (dataset, model, ordering):
          * we have one record per (qid, input index)
          * groupby("ordering").mean() gives you the 'mean over inputs,
            then over queries' (since each query has same NUM_INPUTS).

    This version:
      - ONLY tracks tau and rbo (no nDCG or other metrics).
      - Writes summary + significance (best vs worst per metric).
      - Adds pairwise significance for rel_top vs rel_bottom, and
        selected ordering comparisons, for tau and rbo only.

    Results written to:
      results/<DATASET_SHORT>_<model>/results_order_summary_input_vs_output.txt
    """

    import os
    import json
    from collections import OrderedDict
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    def _load_json(p):
        try:
            with open(p, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def _dedup_preserve(seq):
        """Remove duplicates while preserving order."""
        return list(OrderedDict.fromkeys(seq or []))

    def _sanitize_to_k(out_ids, allowed_ids_set, k):
        """
        Keep only unique items that are in allowed_ids_set, preserving order,
        then (if needed) pad from allowed_ids_set to reach length k.
        """
        seen, filtered = set(), []
        for d in (out_ids or []):
            if d in allowed_ids_set and d not in seen:
                filtered.append(d)
                seen.add(d)
            if len(filtered) == k:
                break
        if len(filtered) < k:
            for d in allowed_ids_set:
                if d not in seen:
                    filtered.append(d)
                    seen.add(d)
                if len(filtered) == k:
                    break
        return filtered

    def _kendall_tau_on_shared(rank_a, rank_b, shared_ids):
        """
        Kendall's τ between rank_a and rank_b restricted to shared_ids.
        rank_a is treated as the baseline ranking.
        """
        a = [d for d in rank_a if d in shared_ids]
        b = [d for d in rank_b if d in shared_ids]
        if len(a) < 2 or len(b) < 2:
            return float("nan")
        pos = {d: i for i, d in enumerate(a)}
        seq_a = [pos[d] for d in b if d in pos]
        if len(seq_a) < 2:
            return float("nan")
        from scipy.stats import kendalltau
        tau_val, _ = kendalltau(seq_a, list(range(len(seq_a))))
        return float(tau_val)

    def _rbo_on_shared(rank_a, rank_b, shared_ids, p):
        """
        RBO between rank_a and rank_b restricted to shared_ids.
        """
        a = [d for d in rank_a if d in shared_ids]
        b = [d for d in rank_b if d in shared_ids]
        a = list(OrderedDict.fromkeys(a))
        b = list(OrderedDict.fromkeys(b))
        if not a or not b:
            return float("nan")
        return RankingSimilarity(a, b).rbo(p=p)

    # ---- main loop ----
    for dataset in DATASETS:
        # ds_config should give you something like (_, topics_name, short_name)
        _, TOPICS_NAME, DATASET_SHORT = ds_config(dataset)

        for model in MODELS:
            results_root = f"results/{DATASET_SHORT}_{model}"
            if not os.path.isdir(results_root):
                print(f"⚠️  Skipping {dataset}×{model}: folder not found.")
                continue

            print(f"\n=== Analyzing (input→output) {dataset} × {model} ===")
            records = []

            # Directories
            shuffle_dir   = os.path.join(results_root, "N10", BASE_MODE)  # half_relevant @ N10
            shuffle_dir20 = os.path.join(results_root, "N20", BASE_MODE)  # half_relevant @ N20 (for shared10)
            order_dirs = {
                "rel_top":    os.path.join(results_root, "order_exp", "rel_top"),
                "rel_bottom": os.path.join(results_root, "order_exp", "rel_bottom"),
                "rel_middle": os.path.join(results_root, "order_exp", "rel_middle"),
                "rel_ends":   os.path.join(results_root, "order_exp", "rel_ends"),
            }

            # ---- SHUFFLE: shared between N10 and N20 (shared10), input vs outputs ----
            if os.path.isdir(shuffle_dir):
                qids = sorted(
                    q for q in os.listdir(shuffle_dir)
                    if os.path.isdir(os.path.join(shuffle_dir, q))
                )
                for qid in tqdm(qids, desc=f"{dataset}_{model} shuffle (input→output)"):
                    for i in range(NUM_INPUTS):
                        f0 = os.path.join(shuffle_dir, qid, f"in{i:02d}_run0.json")
                        f1 = os.path.join(shuffle_dir, qid, f"in{i:02d}_run1.json")
                        jin0, jin1 = _load_json(f0), _load_json(f1)
                        # need both runs for this input
                        if not (jin0 and jin1):
                            continue

                        # Ordered input list at N10
                        in_list_10 = _dedup_preserve(jin0.get("input_docids", []))[:K]
                        in10_set   = set(in_list_10)

                        # N20 inputs for shared@10 (RQ2 style)
                        if os.path.isdir(os.path.join(shuffle_dir20, qid)):
                            f20 = os.path.join(shuffle_dir20, qid, f"in{i:02d}_run0.json")
                            j20 = _load_json(f20)
                            in20_set = set((j20 or {}).get("input_docids", [])[:20])
                        else:
                            in20_set = set()

                        shared10 = (in10_set & in20_set) if in20_set else in10_set

                        # Outputs (sanitize to subset of N10 inputs & length K)
                        allowed = in10_set
                        out0 = _sanitize_to_k(
                            _dedup_preserve(jin0.get("output_docids", [])),
                            allowed, K
                        )
                        out1 = _sanitize_to_k(
                            _dedup_preserve(jin1.get("output_docids", [])),
                            allowed, K
                        )

                        # τ / RBO between INPUT and each output, then mean
                        tau_vals, rbo_vals = [], []

                        tau0 = _kendall_tau_on_shared(in_list_10, out0, shared10)
                        if not np.isnan(tau0):
                            tau_vals.append(tau0)
                        tau1 = _kendall_tau_on_shared(in_list_10, out1, shared10)
                        if not np.isnan(tau1):
                            tau_vals.append(tau1)

                        rbo0 = _rbo_on_shared(in_list_10, out0, shared10, RBO_P)
                        if not np.isnan(rbo0):
                            rbo_vals.append(rbo0)
                        rbo1 = _rbo_on_shared(in_list_10, out1, shared10, RBO_P)
                        if not np.isnan(rbo1):
                            rbo_vals.append(rbo1)

                        tau = float(np.mean(tau_vals)) if tau_vals else float("nan")
                        rbo = float(np.mean(rbo_vals)) if rbo_vals else float("nan")

                        if not np.isnan(tau):
                            records.append(
                                {"ordering": "shuffle", "tau": tau, "rbo": rbo}
                            )

            # ---- ORDERED VARIANTS: input vs outputs on their own N10 inputs ----
            for ordering, root_path in order_dirs.items():
                if not os.path.isdir(root_path):
                    continue
                qids = sorted(
                    q for q in os.listdir(root_path)
                    if os.path.isdir(os.path.join(root_path, q))
                )
                for qid in tqdm(qids, desc=f"{dataset}_{model} {ordering} (input→output)"):
                    for i in range(NUM_INPUTS):
                        f0 = os.path.join(root_path, qid, f"in{i:02d}_run0.json")
                        f1 = os.path.join(root_path, qid, f"in{i:02d}_run1.json")
                        j0, j1 = _load_json(f0), _load_json(f1)
                        if not (j0 and j1):
                            continue

                        # Ordered inputs (these are always K=10 here)
                        in_list = _dedup_preserve(j0.get("input_docids", []))[:K]
                        in_ids  = set(in_list)

                        out0 = _sanitize_to_k(
                            _dedup_preserve(j0.get("output_docids", [])),
                            in_ids, K
                        )
                        out1 = _sanitize_to_k(
                            _dedup_preserve(j1.get("output_docids", [])),
                            in_ids, K
                        )

                        shared = in_ids  # all K inputs

                        # τ / RBO between INPUT and each output, then mean
                        tau_vals, rbo_vals = [], []

                        tau0 = _kendall_tau_on_shared(in_list, out0, shared)
                        if not np.isnan(tau0):
                            tau_vals.append(tau0)
                        tau1 = _kendall_tau_on_shared(in_list, out1, shared)
                        if not np.isnan(tau1):
                            tau_vals.append(tau1)

                        rbo0 = _rbo_on_shared(in_list, out0, shared, RBO_P)
                        if not np.isnan(rbo0):
                            rbo_vals.append(rbo0)
                        rbo1 = _rbo_on_shared(in_list, out1, shared, RBO_P)
                        if not np.isnan(rbo1):
                            rbo_vals.append(rbo1)

                        tau = float(np.mean(tau_vals)) if tau_vals else float("nan")
                        rbo = float(np.mean(rbo_vals)) if rbo_vals else float("nan")

                        if not np.isnan(tau):
                            records.append(
                                {"ordering": ordering, "tau": tau, "rbo": rbo}
                            )

            # ---- Aggregate & write summary ----
            df = pd.DataFrame(records)
            if df.empty:
                print(f"⚠️  No records for {dataset}×{model} (input→output).")
                continue

            # reorder so shuffle is first, then your ORDERINGS
            order_index = ["shuffle"] + list(ORDERINGS)
            summary = (
                df.groupby("ordering")[["tau", "rbo"]]
                .mean()
                .reindex(order_index, fill_value=np.nan)
                .reset_index()
            )

            out_path = os.path.join(
                results_root,
                "results_order_summary_input_vs_output.txt"
            )

            with open(out_path, "w") as out:
                out.write(
                    "Aggregated τ (input→output), "
                    "RBO@0.79 (input→output) per ordering\n"
                )
                out.write("=" * 70 + "\n")
                out.write(f"Dataset: {dataset}\nModel: {model}\n\n")
                out.write(f"{'Ordering':<16} {'τ':>12} {'RBO@0.79':>14}\n")
                out.write("-" * 50 + "\n")
                for _, row in summary.iterrows():
                    if pd.isna(row["tau"]):
                        continue
                    out.write(
                        f"{row['ordering']:<16} "
                        f"{row['tau']:>12.6f} {row['rbo']:>14.6f}\n"
                    )

                # Optionally: add RQ2 shuffle row from previous summary (if available)
                rq2_vals = _read_rq2_half_n10_metrics(results_root)
                if rq2_vals is not None:
                    tau10_rq2, rbo10_rq2, _ndcg_rq2 = rq2_vals
                    out.write(
                        f"{'shuffle (RQ2 N10)':<16} "
                        f"{tau10_rq2:>12.6f} {rbo10_rq2:>14.6f}\n"
                    )

            # --- Significance sections (append; do not overwrite) ---
            with open(out_path, "a") as out:
                out.write("\nSignificance (best vs worst per metric):\n")
                out.write(_concise_sig_line(df, "tau", "tau"))
                out.write(_concise_sig_line(df, "rbo", "rbo"))

                out.write("\nPairwise significance: rel_top vs rel_bottom\n")
                out.write("(Two-sided Mann–Whitney U, Welch t; Cliff's δ reported)\n")

            # rel_top vs rel_bottom for tau & rbo
            for label in ["tau", "rbo"]:
                _append_pairwise_top_vs_bottom(out_path, df, label)

            # selected additional pairwise comparisons, tau & rbo only
            with open(out_path, "a") as out:
                out.write("\nPairwise significance: selected ordering comparisons\n")
                out.write("(Two-sided Mann–Whitney U, Welch t; Cliff's δ reported)\n")

            pair_tests = [
                # rel_top vs others (rel_top vs rel_bottom already logged above too)
                ("rel_top",    "rel_middle"),
                ("rel_top",    "rel_ends"),
                ("rel_top",    "shuffle"),
                ("rel_top",    "rel_bottom"),

                # rel_middle vs rel_bottom, rel_ends, shuffle
                ("rel_middle", "rel_bottom"),
                ("rel_middle", "rel_ends"),
                ("rel_middle", "shuffle"),

                # rel_ends vs rel_bottom, rel_middle, shuffle
                ("rel_ends",   "rel_bottom"),
                ("rel_ends",   "rel_middle"),
                ("rel_ends",   "shuffle"),

                # shuffle vs rel_bottom, rel_middle, rel_ends, rel_top
                ("shuffle",    "rel_bottom"),
                ("shuffle",    "rel_middle"),
                ("shuffle",    "rel_ends"),
                ("shuffle",    "rel_top"),
            ]

            metrics_for_pairs = ["tau", "rbo"]

            for (ord_a, ord_b) in pair_tests:
                for label in metrics_for_pairs:
                    _append_pairwise_ordering(out_path, df, label, ord_a, ord_b)

            print(f"✅ Input→output summary + significance saved to {out_path}")




compute_order_metrics_input_vs_output()  # new definition (input→output)


########################################################################################################################
################################### plot the linees #################################################################
#######################################################################################################################

import os, json, math
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyserini.search import get_qrels


def plot_ndcg_orderings_per_query(dataset_short, model_short,
                                  base_mode="half_relevant",
                                  K=10,
                                  num_inputs=50,
                                  out_path=None):
    """
    Build per-query nDCG for each ordering (shuffle, rel_top, rel_bottom,
    rel_middle, rel_ends) and plot 5 lines over queries sorted by rel_top
    nDCG (ascending).

    Assumes log layout:
      results/{dataset_short}_{model_short}/N10/{base_mode}/{qid}/inXX_run{0,1}.json
      results/{dataset_short}_{model_short}/order_exp/{ordering}/{qid}/inXX_run{0,1}.json

    where each JSON contains: input_docids, output_docids, qid, ...
    """

    ORDERINGS = ["rel_top", "rel_bottom", "rel_middle", "rel_ends"]

    # --------- helpers ---------
    def _topics_name(dshort):
        if dshort == "trec-covid":
            return "beir-v1.0.0-trec-covid-test"
        elif dshort == "nfcorpus":
            return "beir-v1.0.0-nfcorpus-test"
        else:
            raise ValueError(f"Unknown dataset_short: {dshort}")

    def _load_json(p):
        try:
            with open(p, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def _robust_qrels(qrels_raw, qid):
        if qid in qrels_raw:
            return qrels_raw[qid]
        try:
            qi = int(qid)
            if qi in qrels_raw:
                return qrels_raw[qi]
        except Exception:
            pass
        return qrels_raw.get(str(qid), {})

    def dcg_at_k(labels, k):
        return sum(l / math.log2(i + 2) for i, l in enumerate(labels[:k]))

    def ndcg_for_out(out_docids, qrel_dict, k):
        labs = [int(qrel_dict.get(d, 0)) for d in out_docids[:k]]
        dcg = dcg_at_k(labs, k)
        ideal = sorted(labs, reverse=True)
        idcg = dcg_at_k(ideal, k)
        return (dcg / idcg) if idcg > 0 else 0.0

    def _dedup_preserve(seq):
        return list(OrderedDict.fromkeys(seq or []))

    def _sanitize_to_k(out_ids, allowed_ids_set, k):
        seen, filtered = set(), []
        for d in out_ids or []:
            if d in allowed_ids_set and d not in seen:
                filtered.append(d)
                seen.add(d)
            if len(filtered) == k:
                break
        if len(filtered) < k:
            for d in allowed_ids_set:
                if d not in seen:
                    filtered.append(d)
                    seen.add(d)
                if len(filtered) == k:
                    break
        return filtered

    # --------- prepare qrels and paths ---------
    topics = _topics_name(dataset_short)
    qrels = get_qrels(topics)

    results_root = os.path.join("results", f"{dataset_short}_{model_short}")
    if not os.path.isdir(results_root):
        raise FileNotFoundError(f"Results folder not found: {results_root}")

    # collect ndcg per (ordering, qid) over inputs
    ndcg_lists = defaultdict(list)

    # --------- 1) shuffle condition: N10 / base_mode ---------
    shuffle_dir = os.path.join(results_root, "N10", base_mode)
    if os.path.isdir(shuffle_dir):
        qids = sorted(
            q for q in os.listdir(shuffle_dir)
            if os.path.isdir(os.path.join(shuffle_dir, q))
        )
        for qid in qids:
            qrel_dict = _robust_qrels(qrels, qid)
            for i in range(num_inputs):
                f0 = os.path.join(shuffle_dir, qid, f"in{i:02d}_run0.json")
                f1 = os.path.join(shuffle_dir, qid, f"in{i:02d}_run1.json")
                j0, j1 = _load_json(f0), _load_json(f1)
                if not (j0 and j1):
                    continue

                in_ids = set(j0.get("input_docids", [])[:K])
                if not in_ids:
                    continue

                out0 = _sanitize_to_k(_dedup_preserve(j0.get("output_docids", [])), in_ids, K)
                out1 = _sanitize_to_k(_dedup_preserve(j1.get("output_docids", [])), in_ids, K)

                nd0 = ndcg_for_out(out0, qrel_dict, K)
                nd1 = ndcg_for_out(out1, qrel_dict, K)
                nd_mean = 0.5 * (nd0 + nd1)

                ndcg_lists[("shuffle", qid)].append(nd_mean)

    # --------- 2) ordered variants: rel_top, rel_bottom, rel_middle, rel_ends ---------
    for ordering in ORDERINGS:
        root_path = os.path.join(results_root, "order_exp", ordering)
        if not os.path.isdir(root_path):
            continue

        qids = sorted(
            q for q in os.listdir(root_path)
            if os.path.isdir(os.path.join(root_path, q))
        )
        for qid in qids:
            qrel_dict = _robust_qrels(qrels, qid)
            for i in range(num_inputs):
                f0 = os.path.join(root_path, qid, f"in{i:02d}_run0.json")
                f1 = os.path.join(root_path, qid, f"in{i:02d}_run1.json")
                j0, j1 = _load_json(f0), _load_json(f1)
                if not (j0 and j1):
                    continue

                in_ids = set(j0.get("input_docids", [])[:K])
                if not in_ids:
                    continue

                out0 = _sanitize_to_k(_dedup_preserve(j0.get("output_docids", [])), in_ids, K)
                out1 = _sanitize_to_k(_dedup_preserve(j1.get("output_docids", [])), in_ids, K)

                nd0 = ndcg_for_out(out0, qrel_dict, K)
                nd1 = ndcg_for_out(out1, qrel_dict, K)
                nd_mean = 0.5 * (nd0 + nd1)

                ndcg_lists[(ordering, qid)].append(nd_mean)

    # --------- 3) build df_q: one ndcg per (qid, ordering) ---------
    rows = []
    for (ordering, qid), vals in ndcg_lists.items():
        if not vals:
            continue
        rows.append({
            "qid": qid,
            "ordering": ordering,
            "ndcg": float(np.mean(vals)),
        })

    if not rows:
        raise RuntimeError("No nDCG data collected; check your results folders/logs.")

    df_q = pd.DataFrame(rows)

    # --------- 4) sort queries by rel_top nDCG ascending ---------
    rel_top_df = df_q[df_q["ordering"] == "rel_top"].copy()
    if rel_top_df.empty:
        raise ValueError("No rel_top rows in df_q; cannot sort by rel_top nDCG.")

    rel_top_df = rel_top_df.sort_values("ndcg", ascending=True)
    ordered_qids = rel_top_df["qid"].tolist()

    # --------- 5) pivot to get one nDCG sequence per ordering ---------
    pivot = df_q.pivot(index="qid", columns="ordering", values="ndcg")

    x = np.arange(len(ordered_qids))  # 0..N-1
    color_map = {
        "shuffle":     "grey",
        "rel_top":     "forestgreen",
        "rel_bottom":  "crimson",
        "rel_middle":  "royalblue",
        "rel_ends":    "darkorange",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for ordering in ["shuffle"] + ORDERINGS:
        if ordering not in pivot.columns:
            continue
        y = pivot[ordering].reindex(ordered_qids).to_numpy()
        c = color_map.get(ordering, None)

        # 1) light line
        ax.plot(
            x, y,
            linestyle="-",
            linewidth=2,
            color=c,
            alpha=0.4,          # ← make the line lighter
            label=ordering,     # label goes on the line
        )

        # 2) dark markers on top
        ax.plot(
            x, y,
            linestyle="None",
            marker="o",
            markersize=4,
            color=c,            # full opacity = dark
        )


    # x-axis: no tick labels
    ax.set_xticks([])

    # y-axis: fixed from 0.7 to 1.0
    ax.set_ylim(0.7, 1.0)

    ax.set_xlabel("Queries (sorted by rel_top nDCG, ascending)", fontsize=13)
    ax.set_ylabel("nDCG@10", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()

    if out_path is None:
        os.makedirs("figs", exist_ok=True)
        out_path = os.path.join(
            "figs",
            f"{dataset_short}_{model_short}_ndcg_per_query_sorted_reltop.pdf"
        )

    fig.savefig(out_path, bbox_inches="tight")
    print(f"✓ wrote {out_path}")


# plot_ndcg_orderings_per_query("trec-covid", "gpt-3.5")

###########################################################################################################################
############################################ plot the correlation ##########################################################
############################################################################################################################


# ======= style controls for correlation plots =======
PASTEL = {"N5": "#C7EA46", "N10": "#F6BDC7", "N20": "#C1D9E9"}
DARK   = {"N5": "#6E8B3D", "N10": "#A33A4B", "N20": "#3B688A"}

LINEWIDTH  = 2.8
MARKERSIZE = 6.0

AXIS_LABEL_FONTSIZE = 17
TICK_FONTSIZE       = 17
TITLE_FONTSIZE      = 17


def plot_ndcg_ordering_correlations(dataset_short, model_short,
                                    base_mode="half_relevant",
                                    K=10,
                                    num_inputs=50,
                                    out_path=None):
    """
    For each pair of orderings, scatter plot per-query nDCG (ordering A vs ordering B),
    with:
      - grey dotted diagonal y=x
      - solid coloured regression line
      - different colour per subplot (10 total)
      - no 'nDCG@10' axis labels (titles only)

    Uses the same directory structure as plot_ndcg_orderings_per_query.
    """
    import os, json, math
    from collections import OrderedDict, defaultdict

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pyserini.search import get_qrels

    ORDERINGS = ["rel_top", "rel_bottom", "rel_middle", "rel_ends"]

    # ---------- helpers (copied from your ndcg plot) ----------
    def _topics_name(dshort):
        if dshort == "trec-covid":
            return "beir-v1.0.0-trec-covid-test"
        elif dshort == "nfcorpus":
            return "beir-v1.0.0-nfcorpus-test"
        else:
            raise ValueError(f"Unknown dataset_short: {dshort}")

    def _load_json(p):
        try:
            with open(p, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def _robust_qrels(qrels_raw, qid):
        if qid in qrels_raw:
            return qrels_raw[qid]
        try:
            qi = int(qid)
            if qi in qrels_raw:
                return qrels_raw[qi]
        except Exception:
            pass
        return qrels_raw.get(str(qid), {})

    def dcg_at_k(labels, k):
        return sum(l / math.log2(i + 2) for i, l in enumerate(labels[:k]))

    def ndcg_for_out(out_docids, qrel_dict, k):
        labs = [int(qrel_dict.get(d, 0)) for d in out_docids[:k]]
        dcg = dcg_at_k(labs, k)
        ideal = sorted(labs, reverse=True)
        idcg = dcg_at_k(ideal, k)
        return (dcg / idcg) if idcg > 0 else 0.0

    def _dedup_preserve(seq):
        return list(OrderedDict.fromkeys(seq or []))

    def _sanitize_to_k(out_ids, allowed_ids_set, k):
        seen, filtered = set(), []
        for d in out_ids or []:
            if d in allowed_ids_set and d not in seen:
                filtered.append(d)
                seen.add(d)
            if len(filtered) == k:
                break
        if len(filtered) < k:
            for d in allowed_ids_set:
                if d not in seen:
                    filtered.append(d)
                    seen.add(d)
                if len(filtered) == k:
                    break
        return filtered

    # ---------- 1) Build per-query nDCG for each ordering ----------
    topics = _topics_name(dataset_short)
    qrels = get_qrels(topics)

    results_root = os.path.join("results", f"{dataset_short}_{model_short}")
    if not os.path.isdir(results_root):
        raise FileNotFoundError(f"Results folder not found: {results_root}")

    ndcg_lists = defaultdict(list)  # key: (ordering, qid) -> list of ndcgs

    # --- shuffle (N10 / base_mode) ---
    shuffle_dir = os.path.join(results_root, "N10", base_mode)
    if os.path.isdir(shuffle_dir):
        qids = sorted(
            q for q in os.listdir(shuffle_dir)
            if os.path.isdir(os.path.join(shuffle_dir, q))
        )
        for qid in qids:
            qrel_dict = _robust_qrels(qrels, qid)
            for i in range(num_inputs):
                f0 = os.path.join(shuffle_dir, qid, f"in{i:02d}_run0.json")
                f1 = os.path.join(shuffle_dir, qid, f"in{i:02d}_run1.json")
                j0, j1 = _load_json(f0), _load_json(f1)
                if not (j0 and j1):
                    continue

                in_ids = set(j0.get("input_docids", [])[:K])
                if not in_ids:
                    continue

                out0 = _sanitize_to_k(_dedup_preserve(j0.get("output_docids", [])), in_ids, K)
                out1 = _sanitize_to_k(_dedup_preserve(j1.get("output_docids", [])), in_ids, K)

                nd0 = ndcg_for_out(out0, qrel_dict, K)
                nd1 = ndcg_for_out(out1, qrel_dict, K)
                nd_mean = 0.5 * (nd0 + nd1)

                ndcg_lists[("shuffle", qid)].append(nd_mean)

    # --- ordered variants ---
    for ordering in ORDERINGS:
        root_path = os.path.join(results_root, "order_exp", ordering)
        if not os.path.isdir(root_path):
            continue

        qids = sorted(
            q for q in os.listdir(root_path)
            if os.path.isdir(os.path.join(root_path, q))
        )
        for qid in qids:
            qrel_dict = _robust_qrels(qrels, qid)
            for i in range(num_inputs):
                f0 = os.path.join(root_path, qid, f"in{i:02d}_run0.json")
                f1 = os.path.join(root_path, qid, f"in{i:02d}_run1.json")
                j0, j1 = _load_json(f0), _load_json(f1)
                if not (j0 and j1):
                    continue

                in_ids = set(j0.get("input_docids", [])[:K])
                if not in_ids:
                    continue

                out0 = _sanitize_to_k(_dedup_preserve(j0.get("output_docids", [])), in_ids, K)
                out1 = _sanitize_to_k(_dedup_preserve(j1.get("output_docids", [])), in_ids, K)

                nd0 = ndcg_for_out(out0, qrel_dict, K)
                nd1 = ndcg_for_out(out1, qrel_dict, K)
                nd_mean = 0.5 * (nd0 + nd1)

                ndcg_lists[(ordering, qid)].append(nd_mean)

    # build df_q: one ndcg per (qid, ordering)
    rows = []
    for (ordering, qid), vals in ndcg_lists.items():
        if not vals:
            continue
        rows.append({
            "qid": qid,
            "ordering": ordering,
            "ndcg": float(np.mean(vals)),
        })

    if not rows:
        raise RuntimeError("No nDCG data collected; check your results folders/logs.")

    df_q = pd.DataFrame(rows)
    pivot = df_q.pivot(index="qid", columns="ordering", values="ndcg")

    # ---------- 2) Define ordering pairs & colours ----------
    pairs = [
        ("rel_top",   "rel_ends"),
        ("rel_top",   "rel_middle"),
        ("rel_top",   "shuffle"),
        ("rel_top",   "rel_bottom"),
        ("rel_ends",  "rel_middle"),
        ("rel_ends",  "shuffle"),
        ("rel_ends",  "rel_bottom"),
        ("rel_middle","shuffle"),
        ("rel_middle","rel_bottom"),
        ("shuffle",   "rel_bottom"),
    ]

    # 10 distinct colours (one per panel)
    panel_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    # ---------- 3) Make 5×2 grid of scatter plots ----------
    fig, axes = plt.subplots(5, 2, figsize=(10, 16))
    axes = axes.ravel()

    for idx, ((ord_a, ord_b), ax) in enumerate(zip(pairs, axes)):
        if ord_a not in pivot.columns or ord_b not in pivot.columns:
            ax.axis("off")
            continue

        # drop queries where either is NaN
        sub = pivot[[ord_a, ord_b]].dropna()
        if sub.empty:
            ax.axis("off")
            continue

        x = sub[ord_a].to_numpy()
        y = sub[ord_b].to_numpy()

        color = panel_colors[idx]

        # scatter points
        ax.scatter(x, y, s=15, color=color, alpha=0.6, edgecolor="none")

        # diagonal y=x (dotted grey)
        ax.plot([0, 1], [0, 1],
                linestyle=":",
                color="grey",
                linewidth=1.5)

        # regression line (solid colour)
        if len(x) >= 2:
            coef = np.polyfit(x, y, 1)
            x_line = np.linspace(0, 1, 100)
            y_line = coef[0] * x_line + coef[1]
            ax.plot(x_line, y_line,
                    linestyle="-",
                    color=color,
                    linewidth=2.0)

            # Pearson correlation
            r = np.corrcoef(x, y)[0, 1]
            title = f"{ord_a} vs {ord_b} (r={r:.2f})"
        else:
            title = f"{ord_a} vs {ord_b}"

        ax.set_title(title, fontsize=TITLE_FONTSIZE)

        # axis limits fixed to [0,1] (since these are nDCG scores)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)

        # no axis labels "nDCG@10"
        ax.set_xlabel("")
        ax.set_ylabel("")

        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.grid(True, alpha=0.3)

    # Hide any extra axes if pairs < 10 (shouldn't happen, but just in case)
    for j in range(len(pairs), len(axes)):
        axes[j].axis("off")

    fig.tight_layout()

    if out_path is None:
        os.makedirs("figs", exist_ok=True)
        out_path = os.path.join(
            "figs",
            f"{dataset_short}_{model_short}_ndcg_ordering_correlations.pdf"
        )

    fig.savefig(out_path, bbox_inches="tight")
    print(f"✓ wrote {out_path}")


    
# plot_ndcg_ordering_correlations("trec-covid", "gpt-3.5")