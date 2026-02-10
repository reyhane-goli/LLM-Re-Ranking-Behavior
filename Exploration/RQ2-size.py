# unified_experiment.py
# ------------------------------------------------------------
# Flexible experiment runner for LLM re-ranking consistency.
# - Model selection:  "gpt-3.5-turbo"  or  "gpt-4o-mini"
# - Dataset choice:   "covid" or "nfcorpus"
# - Modes:            ["half_relevant", "only_relevant", "only_nonrelevant"]
# - Sizes:            [5, 10, 20]  (incremental growth preserved)
# - Inputs/query:     NUM_INPUTS (default 50)
# - Runs/input:       DETER_RUNS (default 2)
#
# Logs are saved per:
#   results/{DATASET}_{MODEL_SHORT}/N5|N10|N20/{mode}/{qid}/inXX_runY.json
#
# Each log mirrors your existing structure:
#   {
#     "qid": ...,
#     "mode": ...,
#     "size": ...,
#     "input_index": ...,
#     "run": ...,
#     "prompt": ...,
#     "input_docids": [...],
#     "output_docids": [...],
#     "response": ...
#   }
# ------------------------------------------------------------

# ===============================================================
# RQ2-size.py  (updated)
# ---------------------------------------------------------------
# Replaces: only_relevant → single_relevant
#            only_nonrelevant → single_nonrelevant
# Everything else unchanged.
# ===============================================================

import os
import json
import copy
import random
from typing import List, Dict, Tuple

from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels

from rank_gpt import create_permutation_instruction, run_llm, receive_permutation

from scipy.stats import kendalltau
from collections import OrderedDict


# =========================
# Configuration
# =========================
MODEL_NAME = "gpt-4o-mini"      # gpt-3.5-turbo or gpt-4o-mini
DATASET = "nfcorpus"                 # covid or "nfcorpus"
MODES = ["half_relevant", "single_relevant", "single_nonrelevant"]
SIZES = [5, 10, 20]
NUM_INPUTS = 50
DETER_RUNS = 2
SEED = 42

API_KEY = ""

# =========================
# Derived config
# =========================
if DATASET.lower() == "covid":
    INDEX_NAME = "beir-v1.0.0-trec-covid.flat"
    TOPICS_NAME = "beir-v1.0.0-trec-covid-test"
    DATASET_SHORT = "trec-covid"
elif DATASET.lower() == "nfcorpus":
    INDEX_NAME = "beir-v1.0.0-nfcorpus.flat"
    TOPICS_NAME = "beir-v1.0.0-nfcorpus-test"
    DATASET_SHORT = "nfcorpus"
else:
    raise ValueError("DATASET must be 'covid' or 'nfcorpus'.")

MODEL_SHORT = "gpt-3.5" if "3.5" in MODEL_NAME else "gpt-4o-mini"
RESULTS_ROOT = f"results/{DATASET_SHORT}_{MODEL_SHORT}"

random.seed(SEED)

# =========================
# Helpers
# =========================
def ensure_dir(path): os.makedirs(path, exist_ok=True)

# def sample_pool(pool: List[str], k: int) -> List[str]:
#     if len(pool) >= k: return random.sample(pool, k)
#     return random.choices(pool, k=k)

def sample_pool(pool, k):
    """Prefer unique sampling; if pool is small, take all uniques then top-up with replacement."""
    if k <= 0:
        return []
    if len(pool) >= k:
        return random.sample(pool, k)
    # Not enough unique candidates; take everything, then top up with replacement
    taken = list(pool)
    need_more = k - len(taken)
    if need_more > 0 and len(pool) > 0:
        taken += random.choices(pool, k=need_more)
    return taken

import random

def sample_exact_unique(pool, k, exclude=None, fallback_with_replacement=True, label_when_short=""):
    """
    Return exactly k **unique** ids from pool, excluding any in `exclude`.
    If pool is short and fallback_with_replacement is True, fill the remainder
    by sampling **with replacement** (but keep existing uniques). This guarantees length k.
    """
    if exclude is None:
        exclude = set()
    else:
        exclude = set(exclude)

    uniq = [d for d in pool if d not in exclude]
    if len(uniq) >= k:
        return random.sample(uniq, k)

    chosen = list(uniq)
    need = k - len(chosen)
    if need <= 0:
        return chosen

    if not fallback_with_replacement:
        # hard stop: not enough unique docs
        raise ValueError(f"[{label_when_short}] Not enough unique items to sample {k}; got {len(uniq)}.")

    # Fill remainder with replacement (guarantee size k)
    # If uniq is empty, use the *raw* pool to avoid empty sampling errors.
    base = uniq if len(uniq) > 0 else (pool if len(pool) > 0 else [])
    if len(base) == 0:
        # nothing to sample at all – return what we have (short) and let caller decide
        return chosen
    chosen += random.choices(base, k=need)
    return chosen

# def build_rel_nonrel_pools(qid, qrels, searcher, topics, top_k_for_nonrel=5000):
#     rel_docs = [d for d, r in qrels[qid].items() if int(r) > 0 and searcher.doc(d)]
#     explicit_nonrel = [d for d, r in qrels[qid].items() if int(r) == 0 and searcher.doc(d)]
#     if explicit_nonrel:
#         nonrel_docs = explicit_nonrel
#     else:
#         query_text = topics[qid]["title"]
#         hits = searcher.search(query_text, k=top_k_for_nonrel)
#         cand = [h.docid for h in hits]
#         nonrel_docs = [d for d in cand if d not in rel_docs and searcher.doc(d)]
        

#     return rel_docs, nonrel_docs

########################
def _topics_title(topics, qid):
    # robust: topics keys can be int or str depending on dataset
    if qid in topics: 
        return topics[qid]["title"]
    try:
        return topics[int(qid)]["title"]
    except:
        return topics[str(qid)]["title"]

def _expand_candidates_from_bm25(searcher, query_text, rel_blocklist, need, already, k=20000):
    """
    Pull up to `need` more docids from a large BM25 pool, excluding anything in rel_blocklist
    and anything we already have in `already`. Returns an extended list (dedup preserved).
    """
    if need <= 0:
        return already
    hits = searcher.search(query_text, k=k)
    for h in hits:
        d = h.docid
        if d in rel_blocklist: 
            continue
        if d in already:
            continue
        already.append(d)
        if len(already) >= need:
            break
    return already

def _max_requirements_from_modes_sizes(modes, sizes):
    """
    Compute worst-case counts needed by any mode for the largest k in `sizes`.
    - For `single_relevant`: need (k-1) nonrelevant.
    - For `single_nonrelevant`: need (k-1) relevant.
    - For `half_relevant`: roughly k//2 each; we set a safe margin.
    """
    Kmax = max(sizes) if sizes else 0
    need_nonrel = 0
    need_rel    = 0
    if "single_relevant" in modes:
        need_nonrel = max(need_nonrel, Kmax - 1)
    if "single_nonrelevant" in modes:
        need_rel = max(need_rel, Kmax - 1)
    if "half_relevant" in modes:
        need_rel    = max(need_rel,    (Kmax + 1)//2)
        need_nonrel = max(need_nonrel,  Kmax//2)
    return need_rel, need_nonrel


# ---- helpers to robustly pick docs without crashing ----
def make_more_nonrel_provider(qid, topics, searcher, qrels, topk=20000):
    """Return a callable that yields more candidate non-relevant docids for this qid."""
    rel_set = {d for d, r in qrels[qid].items() if int(r) > 0}
    query_text = topics[qid]["title"]
    hits = searcher.search(query_text, k=topk)
    bm25_pool = [h.docid for h in hits if h.docid not in rel_set]
    def supply():
        return bm25_pool
    return supply

def pick_from(pool, k, exclude):
    # sample up to k items from pool excluding items in exclude; with replacement if needed
    choices = [d for d in pool if d not in exclude]
    if len(choices) >= k:
        return random.sample(choices, k)
    # not enough unique; take all unique + top up with replacement
    out = list(choices)
    if k > len(out):
        # allow repeats, but don’t repeat within this call if possible
        repeats = random.choices(pool, k=k-len(out))
        out += repeats
    return out

#########

from pyserini.search.lucene import LuceneSearcher

def build_rel_nonrel_pools(qid, qrels, searcher: LuceneSearcher, topics, top_k_for_nonrel=5000):
    """
    rel_docs: qrels[qid] > 0 and exists in index
    nonrel_docs: first try qrels[qid] == 0; if empty, pull BM25 and exclude rel set.
    """
    # robust qrels lookup (qid can be str or int)
    qrel_dict = qrels.get(qid, {})
    if not qrel_dict and isinstance(qid, str):
        try: qrel_dict = qrels.get(int(qid), {})
        except: pass
    if not qrel_dict and isinstance(qid, int):
        qrel_dict = qrels.get(str(qid), {})

    rel_docs = [d for d, r in qrel_dict.items() if int(r) > 0 and searcher.doc(d) is not None]
    explicit_non = [d for d, r in qrel_dict.items() if int(r) == 0 and searcher.doc(d) is not None]

    if explicit_non:
        nonrel_docs = explicit_non
    else:
        # Build nonrelevant via BM25 minus rel set
        query_text = topics[str(qid)]["title"] if str(qid) in topics else topics[int(qid)]["title"]
        hits = searcher.search(query_text, k=top_k_for_nonrel)
        cand = [h.docid for h in hits]
        rel_set = set(rel_docs)
        nonrel_docs = [d for d in cand if d not in rel_set and searcher.doc(d) is not None]

    return rel_docs, nonrel_docs


def choose_queries(dataset, topics, qrels, searcher, desired_n=50):
    if dataset == "covid":
        qids_sorted = sorted(topics.keys(), key=lambda x: int(x))
    else:
        qids_sorted = sorted(topics.keys())
    if dataset == "covid":
        return qids_sorted
    eligible = []
    for qid in qids_sorted:
        rel_docs = [d for d, r in qrels[qid].items() if int(r) > 0 and searcher.doc(d)]
        if len(rel_docs) >= 20:
            eligible.append(qid)
    if len(eligible) < desired_n:
        raise RuntimeError(f"Not enough NF-Corpus queries with >=20 relevant docs. Found {len(eligible)}.")
    return random.sample(eligible, desired_n)

def build_k5_half_relevant(rel_docs, nonrel_docs):
    if random.random() < 0.5:
        need_rel, need_non = 3, 2
    else:
        need_rel, need_non = 2, 3
    combo = sample_pool(rel_docs, need_rel) + sample_pool(nonrel_docs, need_non)
    random.shuffle(combo)
    return combo

# def expand_preserving(current, target_size, rel_docs, nonrel_docs, mode):
#     if len(current) > target_size:
#         raise ValueError("Current list is larger than target size; cannot shrink.")
#     if len(current) == target_size:
#         return list(current)

#     if mode == "single_relevant":
#         rel_part = sample_pool(rel_docs, 1)
#         non_part = sample_pool(nonrel_docs, target_size - 1)
#         out = rel_part + non_part
#         random.shuffle(out)
#         return out

#     if mode == "single_nonrelevant":
#         non_part = sample_pool(nonrel_docs, 1)
#         rel_part = sample_pool(rel_docs, target_size - 1)
#         out = rel_part + non_part
#         random.shuffle(out)
#         return out

#     if mode == "half_relevant":
#         cur_rel = [d for d in current if d in rel_docs]
#         cur_non = [d for d in current if d in nonrel_docs]
#         desired_rel = target_size // 2
#         desired_non = target_size - desired_rel
#         add_rel = max(0, desired_rel - len(cur_rel))
#         add_non = max(0, desired_non - len(cur_non))
#         added = sample_pool([d for d in rel_docs if d not in current], add_rel) + \
#                 sample_pool([d for d in nonrel_docs if d not in current], add_non)
#         out = current + added
#         random.shuffle(out)
#         return out
#     return list(current)




# def expand_preserving(current, target_size, rel_docs, nonrel_docs, mode):
#     """
#     Extend `current` to `target_size` while guaranteeing:
#       - single_relevant: exactly 1 relevant overall
#       - single_nonrelevant: exactly 1 non-relevant overall
#       - half_relevant: ~50/50 split (your existing behavior)
#     Never removes ids already in `current`; avoids duplicates.
#     """
#     if len(current) > target_size:
#         raise ValueError("Current list is larger than target size; cannot shrink.")
#     if len(current) == target_size:
#         return list(current)

#     cur_set = set(current)
#     extra_needed = target_size - len(current)

#     # helper to sample without introducing duplicates
#     def pick_from(pool, k):
#         # prefer unseen
#         unseen = [d for d in pool if d not in cur_set]
#         if len(unseen) >= k:
#             chosen = random.sample(unseen, k)
#         else:
#             # if pool is small, allow repeats in sampling but dedupe against cur_set
#             chosen = unseen + random.sample(pool, max(0, k - len(unseen)))
#             # still ensure uniqueness overall
#             chosen = [d for d in chosen if d not in cur_set]
#             if len(chosen) < k:
#                 # final fallback: fill from anything not yet chosen until we hit k
#                 for d in pool:
#                     if d not in cur_set and d not in chosen:
#                         chosen.append(d)
#                     if len(chosen) == k:
#                         break
#         return chosen[:k]

#     if mode == "single_relevant":
#         cur_rel = [d for d in current if d in rel_docs]
#         cur_non = [d for d in current if d in nonrel_docs]

#         if len(cur_rel) == 0:
#             # ensure exactly 1 relevant by adding one now, then fill the rest with nonrelevant
#             add_rel = pick_from(rel_docs, 1)
#             cur_set.update(add_rel)
#             current = current + add_rel
#             extra_needed -= 1
#         elif len(cur_rel) > 1:
#             # cannot fix here without removal; warn but still extend only with nonrelevant
#             pass

#         # fill remaining with nonrelevant
#         if extra_needed > 0:
#             add_non = pick_from(nonrel_docs, extra_needed)
#             current = current + add_non

#         random.shuffle(current)
#         return current

#     if mode == "single_nonrelevant":
#         cur_non = [d for d in current if d in nonrel_docs]
#         cur_rel = [d for d in current if d in rel_docs]

#         if len(cur_non) == 0:
#             add_non = pick_from(nonrel_docs, 1)
#             cur_set.update(add_non)
#             current = current + add_non
#             extra_needed -= 1
#         elif len(cur_non) > 1:
#             # cannot fix without removal; warn but still extend only with relevant
#             pass

#         if extra_needed > 0:
#             add_rel = pick_from(rel_docs, extra_needed)
#             current = current + add_rel

#         random.shuffle(current)
#         return current

#     # half_relevant (unchanged, but ensure true extension)
#     if mode == "half_relevant":
#         cur_rel = [d for d in current if d in rel_docs]
#         cur_non = [d for d in current if d in nonrel_docs]
#         desired_rel = target_size // 2
#         desired_non = target_size - desired_rel

#         add_rel = max(0, desired_rel - len(cur_rel))
#         add_non = max(0, desired_non - len(cur_non))

#         added = []
#         if add_rel > 0: added += pick_from(rel_docs, add_rel)
#         if add_non > 0: added += pick_from(nonrel_docs, add_non)
#         if len(added) < extra_needed:
#             # fill any residual from the larger remaining pool
#             pool = rel_docs if len(rel_docs) > len(nonrel_docs) else nonrel_docs
#             added += pick_from(pool, extra_needed - len(added))

#         out = current + added
#         random.shuffle(out)
#         return out

#     return list(current)

def expand_preserving(current, target_size, rel_docs, nonrel_docs, mode):
    """
    Strictly PRESERVE current order and IDs; only append new IDs to reach target_size.
    - single_relevant: ensure exactly 1 relevant overall, rest nonrelevant
    - single_nonrelevant: ensure exactly 1 nonrelevant overall, rest relevant
    - half_relevant: aim for target_size//2 relevant (±1 tolerance handled by top-up)
    """
    cur = list(current)
    if len(cur) > target_size:
        raise ValueError("Current list is larger than target size; cannot shrink.")
    if len(cur) == target_size:
        return cur

    cur_set = set(cur)
    cur_rel = [d for d in cur if d in rel_docs]
    cur_non = [d for d in cur if d in nonrel_docs]

    if mode == "single_relevant":
        # We want exactly 1 relevant in the FINAL list.
        # If current already has >1 relevant, keep the first, and treat extras as fixed (can’t remove),
        # so we just top up with nonrelevant; you’ll catch any “not exactly 1” in diagnostics.
        need_total = target_size - len(cur)
        # Prefer to add only nonrelevant; if none available, fall back to any doc not in cur
        add_non = need_total
        added = pick_from(nonrel_docs, add_non, exclude=cur_set)
        return cur + added

    if mode == "single_nonrelevant":
        # Exactly 1 nonrelevant in FINAL list; top up with relevant
        need_total = target_size - len(cur)
        add_rel = need_total
        added = pick_from(rel_docs, add_rel, exclude=cur_set)
        return cur + added

    # half_relevant: target ~ 50/50
    desired_rel = target_size // 2
    desired_non = target_size - desired_rel
    add_rel = max(0, desired_rel - len(cur_rel))
    add_non = max(0, desired_non - len(cur_non))
    added = []
    if add_rel > 0:
        added += pick_from(rel_docs, add_rel, exclude=cur_set | set(added))
    if add_non > 0:
        added += pick_from(nonrel_docs, add_non, exclude=cur_set | set(added))
    # If still short (e.g., pools too small), top up from the bigger available pool
    while len(cur) + len(added) < target_size:
        # choose whichever pool has more candidates left
        left_rel = [d for d in rel_docs if d not in (cur_set | set(added))]
        left_non = [d for d in nonrel_docs if d not in (cur_set | set(added))]
        if len(left_rel) >= len(left_non):
            added += pick_from(rel_docs, 1, exclude=cur_set | set(added))
        else:
            added += pick_from(nonrel_docs, 1, exclude=cur_set | set(added))
    return cur + added


# ---- Add this helper somewhere near your other helpers ----
def print_comp(mode: str, size: int, inp_idx: int,
               docids: list[str],
               rel_pool: list[str],
               nonrel_pool: list[str]) -> None:
    """Print how many relevant / non-relevant (and unknown) are in docids."""
    rel_set = set(rel_pool)
    non_set = set(nonrel_pool)
    r = sum(1 for d in docids if d in rel_set)
    n = sum(1 for d in docids if d in non_set)
    u = len(docids) - r - n  # should be 0; if not, they weren't in either pool
    print(f"[{mode}] N{size:>2} input={inp_idx:02d} -> rel={r} nonrel={n}" + (f" unknown={u}" if u else ""))

    
# def build_inputs_for_query(qid, topics, qrels, searcher, modes, sizes, num_inputs):
    
# #     rel_docs, nonrel_docs = build_rel_nonrel_pools(qid, qrels, searcher, topics)

#     # NEW: compute worst-case requirements once
#     need_rel_min, need_nonrel_min = _max_requirements_from_modes_sizes(modes, sizes)

#     # Get pools (with NF-Corpus fallback if needed)
#     rel_docs, nonrel_docs = build_rel_nonrel_pools(
#         qid, qrels, searcher, topics,
#         need_rel_min=need_rel_min,
#         need_nonrel_min=need_nonrel_min
#     )
    

    
#     inputs = {m: {s: {} for s in sizes} for m in modes}
#     for mode in modes:
#         for inp_idx in range(num_inputs):
#             # base 5
#             if 5 in sizes:
#                 if mode == "single_relevant":
#                     k5_docids = sample_pool(rel_docs, 1) + sample_pool(nonrel_docs, 4)
#                 elif mode == "single_nonrelevant":
#                     k5_docids = sample_pool(nonrel_docs, 1) + sample_pool(rel_docs, 4)
#                 elif mode == "half_relevant":
#                     k5_docids = build_k5_half_relevant(rel_docs, nonrel_docs)
#                 else:
#                     k5_docids = []
#                 random.shuffle(k5_docids)
#             else:
#                 k5_docids = []

#             #check
# #             print_comp(mode, 5, inp_idx, k5_docids, rel_docs, nonrel_docs)
                
#             # expand
#             k10_docids = expand_preserving(k5_docids, 10, rel_docs, nonrel_docs, mode) if 10 in sizes else []
#             #check
# #             print_comp(mode, 10, inp_idx, k10_docids, rel_docs, nonrel_docs)
            
           
#             base = k10_docids if k10_docids else k5_docids
#             k20_docids = expand_preserving(base, 20, rel_docs, nonrel_docs, mode) if 20 in sizes else []
#             #check
# #             print_comp(mode, 20, inp_idx, k20_docids, rel_docs, nonrel_docs)
            
#             for size, docids in [(5, k5_docids), (10, k10_docids), (20, k20_docids)]:
#                 if size not in sizes or not docids:
#                     continue
#                 hits = []
#                 for d in docids:
#                     dobj = searcher.doc(d)
#                     if not dobj:
#                         continue
#                     hits.append({"docid": d, "content": dobj.raw()})
#                 inputs[mode][size][inp_idx] = hits
#     return inputs


def build_inputs_for_query(qid, topics, qrels, searcher, modes, sizes, num_inputs):
    rel_docs, nonrel_docs = build_rel_nonrel_pools(qid, qrels, searcher, topics)
    supply_nonrel = make_more_nonrel_provider(qid, topics, searcher, qrels)

    inputs = {m: {s: {} for s in sizes} for m in modes}
    for mode in modes:
        for inp_idx in range(num_inputs):
            # base 5
            # base 5
            if 5 in sizes:
                if mode == "single_relevant":
                    k5_docids = sample_exact_unique(rel_docs, 1, label_when_short="N5 single_relevant/rel") + \
                                sample_exact_unique(nonrel_docs, 4, exclude=None, label_when_short="N5 single_relevant/non")
                elif mode == "single_nonrelevant":
                    k5_docids = sample_exact_unique(nonrel_docs, 1, label_when_short="N5 single_nonrelevant/non") + \
                                sample_exact_unique(rel_docs, 4, exclude=None, label_when_short="N5 single_nonrelevant/rel")
                elif mode == "half_relevant":
                    k5_docids = build_k5_half_relevant(rel_docs, nonrel_docs)
                else:
                    k5_docids = []
                random.shuffle(k5_docids)
            else:
                k5_docids = []


            # expand to 10/20 (preserving)
            # expand to 10
            k10_docids = expand_preserving(k5_docids, 10, rel_docs, nonrel_docs, mode) if 10 in sizes else []

            # expand to 20 from the 10 list (NOT from scratch)
            base = k10_docids if k10_docids else k5_docids
            k20_docids = expand_preserving(base, 20, rel_docs, nonrel_docs, mode) if 20 in sizes else []


            # write hits ...
            for size, docids in [(5, k5_docids), (10, k10_docids), (20, k20_docids)]:
                if size not in sizes or not docids:
                    continue
                hits = []
                for d in docids:
                    dobj = searcher.doc(d)
                    if not dobj:
                        continue
                    hits.append({"docid": d, "content": dobj.raw()})
                inputs[mode][size][inp_idx] = hits
    return inputs




def run_and_log_for_query(qid, query_text, inputs, results_root, model_name, api_key, deter_runs):
    for mode, size_map in inputs.items():
        for size, inp_map in size_map.items():
            qdir = os.path.join(results_root, f"N{size}", mode, str(qid))
            ensure_dir(qdir)
            for inp_idx, hits in inp_map.items():
                for run_idx in range(deter_runs):
                    item = {"query": query_text, "hits": copy.deepcopy(hits)}
                    msgs = create_permutation_instruction(item, 0, len(hits), model_name=model_name)
                    resp = run_llm(msgs, api_key=api_key, model_name=model_name)
                    out = receive_permutation(item, resp, 0, len(hits))
                    log = {
                        "qid": qid, "mode": mode, "size": size,
                        "input_index": inp_idx, "run": run_idx,
                        "prompt": msgs,
                        "input_docids": [h["docid"] for h in hits],
                        "output_docids": [h["docid"] for h in out["hits"][:len(hits)]],
                        "response": resp
                    }
                    with open(os.path.join(qdir, f"in{inp_idx:02d}_run{run_idx}.json"), "w") as f:
                        json.dump(log, f, indent=2)



# ===== Resume-safe helpers =====

import os, json

def _expected_qid_paths(results_root, qid, modes, sizes, num_inputs, deter_runs):
    """Yield every expected log file path for this qid across all modes/sizes/inputs/runs."""
    for size in sizes:
        for mode in modes:
            qdir = os.path.join(results_root, f"N{size}", mode, str(qid))
            for inp_idx in range(num_inputs):
                for run_idx in range(deter_runs):
                    yield os.path.join(qdir, f"in{inp_idx:02d}_run{run_idx}.json")

def is_qid_complete(results_root, qid, modes, sizes, num_inputs, deter_runs):
    """
    Strict completeness: all expected files exist.
    If you want a lighter 'marker' check, see is_qid_marked_complete() below.
    """
    for p in _expected_qid_paths(results_root, qid, modes, sizes, num_inputs, deter_runs):
        if not os.path.exists(p):
            return False
    return True

def is_qid_marked_complete(results_root, qid, modes, sizes):
    """
    Faster, looser check: consider a QID complete if for the *largest size*
    every mode has at least one run file present. Good when you just need
    to avoid rerunning done queries without validating every input/run.
    """
    largest = max(sizes)
    for mode in modes:
        qdir = os.path.join(results_root, f"N{largest}", mode, str(qid))
        if not os.path.isdir(qdir):
            return False
        # look for any in??_run1.json (or run0) as a marker
        has_any = any(
            name.endswith("_run0.json") or name.endswith("_run1.json")
            for name in os.listdir(qdir)
            if name.startswith("in") and name.endswith(".json")
        )
        if not has_any:
            return False
    return True

# ---------- Helpers for robust qrels + counting ----------
def get_qrel_dict_robust(qrels, qid):
    if qid in qrels:
        return qrels[qid]
    try:
        return qrels[int(qid)]
    except Exception:
        return qrels.get(str(qid), {})

def count_rel(docids, qrel_dict):
    return sum(1 for d in docids if int(qrel_dict.get(d, 0)) > 0)

def extract_docids_from_hits(hits_list):
    # hits_list is dict: inputs[mode][size][inp_idx] -> [{'docid':..., 'content':...}, ...]
    return [h["docid"] for h in hits_list if "docid" in h]

def validate_inputs_for_query(qid, inputs, qrels, modes, sizes):
    """
    Prints diagnostics only when a QID passes both checks.
    Returns True if (a) composition OK for all present sizes and (b) N5 ⊆ N10 ⊆ N20
    for every input index that exists; otherwise returns False and prints violations.
    """
    ok = True
    qrel_dict = get_qrel_dict_robust(qrels, qid)

    # 1) composition checks
    for mode in modes:
        for size in sizes:
            inp_map = inputs.get(mode, {}).get(size, {})
            if not inp_map:
                continue
            for inp_idx, hits in inp_map.items():
                docids = extract_docids_from_hits(hits)
                rel = count_rel(docids, qrel_dict)
                non = len(docids) - rel

                if mode == "single_relevant":
                    if not (rel == 1 and non == size - 1):
                        print(f"[BAD] Q{qid} {mode} N{size} in={inp_idx:02d}: "
                              f"expected 1 rel/{size-1} non, got {rel} rel/{non} non")
                        ok = False
                elif mode == "single_nonrelevant":
                    if not (rel == size - 1 and non == 1):
                        print(f"[BAD] Q{qid} {mode} N{size} in={inp_idx:02d}: "
                              f"expected {size-1} rel/1 non, got {rel} rel/{non} non")
                        ok = False
                elif mode == "half_relevant":
                    # strict halves for 10/20; allow {2,3} vs {3,2} for 5
                    if size == 5:
                        if not (rel in (2, 3) and non in (2, 3) and rel + non == 5):
                            print(f"[BAD] Q{qid} {mode} N5 in={inp_idx:02d}: "
                                  f"expected 2/3 split, got {rel} rel/{non} non")
                            ok = False
                    elif size == 10 and (rel != 5 or non != 5):
                        print(f"[BAD] Q{qid} {mode} N10 in={inp_idx:02d}: "
                              f"expected 5 rel/5 non, got {rel} rel/{non} non")
                        ok = False
                    elif size == 20 and (rel != 10 or non != 10):
                        print(f"[BAD] Q{qid} {mode} N20 in={inp_idx:02d}: "
                              f"expected 10 rel/10 non, got {rel} rel/{non} non")
                        ok = False

    # 2) subset checks (only print positives like you asked)
    #    We will still set ok=False if subset fails, but only *print* the QIDs that pass.
    all_passed_subset = True
    for mode in modes:
        m = inputs.get(mode, {})
        map5  = m.get(5, {})
        map10 = m.get(10, {})
        map20 = m.get(20, {})

        # Work across *common* input indices we actually have
        common_idxs = set(map5.keys()) | set(map10.keys()) | set(map20.keys())
        for inp_idx in sorted(common_idxs):
            s5  = set(extract_docids_from_hits(map5.get(inp_idx, [])))  if inp_idx in map5  else None
            s10 = set(extract_docids_from_hits(map10.get(inp_idx, []))) if inp_idx in map10 else None
            s20 = set(extract_docids_from_hits(map20.get(inp_idx, []))) if inp_idx in map20 else None

            # Need all three sizes present to check the full chain
            if s5 is None or s10 is None or s20 is None:
                continue

            cond = (s5.issubset(s10) and s10.issubset(s20))
            if cond:
                # You said: only print those that meet the condition
                print(f"[OK SUBSET] Q{qid} {mode} in={inp_idx:02d}: "
                      f"N5⊆N10⊆N20 (|N5|={len(s5)}, |N10|={len(s10)}, |N20|={len(s20)})")
            else:
                all_passed_subset = False

    if not all_passed_subset:
        ok = False
    return ok

# ===== DIAGNOSTIC: per-size relevance counts + subset chain =====
from typing import Dict, List
import os, json

def count_rel_nonrel(docids, qrel_dict):
    rel = sum(1 for d in docids if int(qrel_dict.get(d, 0)) > 0)
    return rel, len(docids) - rel

def robust_qrel_dict(qrels, qid):
    if qid in qrels: return qrels[qid]
    try:
        return qrels[int(qid)]
    except:
        return qrels.get(str(qid), {})

def validate_inputs_for_qid(qid, inputs_map, qrels, modes=("single_relevant","single_nonrelevant","half_relevant")):
    qrel_dict = robust_qrel_dict(qrels, qid)
    for mode in modes:
        sizes_present = sorted([s for s in inputs_map.get(mode, {}) if inputs_map[mode][s]], key=int)
        for inp_idx, _ in inputs_map.get(mode, {}).get(5, {}).items():
            n5  = [h["docid"] for h in inputs_map[mode].get(5,  {}).get(inp_idx, [])]
            n10 = [h["docid"] for h in inputs_map[mode].get(10, {}).get(inp_idx, [])]
            n20 = [h["docid"] for h in inputs_map[mode].get(20, {}).get(inp_idx, [])]

            # Only proceed if all three exist
            if not (n5 and n10 and n20):
                continue

            ok_subset = set(n5).issubset(n10) and set(n10).issubset(n20)
            r5,  nr5  = count_rel_nonrel(n5,  qrel_dict)
            r10, nr10 = count_rel_nonrel(n10, qrel_dict)
            r20, nr20 = count_rel_nonrel(n20, qrel_dict)

            # Contract checks
            if mode == "single_relevant":
                bad = not (r5==1 and r10==1 and r20==1)
            elif mode == "single_nonrelevant":
                bad = not (nr5==1 and nr10==1 and nr20==1)
            else:
                # half_relevant (~50/50)
                bad = not (abs(r10 - 5) <= 1 and abs(r20 - 10) <= 1)

            if ok_subset:
                print(f"[OK SUBSET] Q{qid} {mode} in={inp_idx:02d}: "
                      f"N5⊆N10⊆N20 | N5(rel,non)={r5},{nr5} | N10={r10},{nr10} | N20={r20},{nr20}")
                if bad:
                    print(f"  [WARN CONTRACT] counts don’t match {mode} expectations for Q{qid} in={inp_idx:02d}")
            else:
                print(f"[WARN SUBSET] Q{qid} {mode} in={inp_idx:02d}: subset chain broken")
                print(f"  N5={len(n5)} N10={len(n10)} N20={len(n20)}")

            
# =========================
# Main
# =========================
def main():
    print(f"=== Running with model={MODEL_NAME}, dataset={DATASET_SHORT}")
    ensure_dir(RESULTS_ROOT)
    topics = get_topics(TOPICS_NAME)
    qrels = get_qrels(TOPICS_NAME)

    searcher = LuceneSearcher.from_prebuilt_index(INDEX_NAME)
    qids = choose_queries(DATASET_SHORT, topics, qrels, searcher, desired_n=50)
    
    
    
#     remaining_qids = [
#     q for q in qids
#     if not is_qid_marked_complete(RESULTS_ROOT, q, MODES, SIZES)
#     ]

#     print(f"Total QIDs: {len(qids)} | Already done: {len(qids) - len(remaining_qids)} | To process: {len(remaining_qids)}")
    
    for qid in tqdm(qids, desc="Queries"):
        query_text = topics[qid]["title"]
        inputs = build_inputs_for_query(qid, topics, qrels, searcher, MODES, SIZES, NUM_INPUTS)
        
        # after inputs = build_inputs_for_query(...)
#         validate_inputs_for_qid(qid, inputs, qrels, modes=MODES)

        
        run_and_log_for_query(qid, query_text, inputs, RESULTS_ROOT, MODEL_NAME, API_KEY, DETER_RUNS)
    print("\n✅ Done! Logs under:", RESULTS_ROOT)

# if __name__ == "__main__":
#     main()

    
############################################################################################################    
########################### generating the RBo, NDCG and Tau results ########################################
################################################################################################################

## ===============================================================
#  Multi-dataset × Multi-model log analysis
# ===============================================================


# ===============================================================
# Function: compute_all_metrics_shared_by_ids()
# ===============================================================
# τ and RBO are computed using the ACTUAL SHARED DOC IDS across sizes:
#   shared@5  = intersection of N5, N10, N20 input_docids
#   shared@10 = intersection of N10, N20  input_docids
#   shared@20 = N20 input_docids
# RBO p-values: p5=0.63, p10=0.79, p20=0.89
# nDCG@k uses the full list of the CURRENT size (k=5/10/20).
# Output per dataset×model: results_summary_shared_ids.txt
# ===============================================================

############################################################################################################    
########################### v1 - generating the RBO, NDCG and Tau results ########################################
############################################################################################################

# ===============================================================
#  Multi-dataset × Multi-model log analysis
# ===============================================================

import os, json, numpy as np, pandas as pd
from math import log2
from scipy.stats import kendalltau
from rbo import RankingSimilarity
from tqdm import tqdm
from pyserini.search import get_qrels

def compute_all_metrics_shared_by_ids():
    DATASETS = ["trec-covid", "nfcorpus"]
    MODELS   = ["gpt-3.5", "gpt-4o-mini"]
    MODES    = ["half_relevant", "single_relevant", "single_nonrelevant"]
    SIZES    = [5, 10, 20]
    NUM_INPUTS = 50

    RBO_P = {5: 0.63, 10: 0.79, 20: 0.89}

    # ---------- helpers ----------
    def dcg_at_k(rels, k):
        return sum(rel / log2(i + 2) for i, rel in enumerate(rels[:k]))

    def dcg_for_out(out_docs, qrel_dict, k):
        """Raw DCG@k (no normalization)."""
        labels = [int(qrel_dict.get(d, 0)) for d in out_docs[:k]]
        return dcg_at_k(labels, k)

    def ndcg_for_out(out_docs, qrel_dict, k):
        labels = [int(qrel_dict.get(d, 0)) for d in out_docs[:k]]
        dcg = dcg_at_k(labels, k)
        ideal = sorted(labels, reverse=True)
        idcg = dcg_at_k(ideal, k)
        return dcg / idcg if idcg > 0 else 0.0


    def rbo_score_safe(a_ids, b_ids, p):
        """
        RBO library asserts unique elements; also guard short lists.
        De-duplicate while preserving order. Return NaN if too short.
        """
        a = list(OrderedDict.fromkeys(a_ids or []))
        b = list(OrderedDict.fromkeys(b_ids or []))
        if len(a) < 2 or len(b) < 2:
            return float('nan')
        try:
            return RankingSimilarity(a, b).rbo(p=p)
        except AssertionError:
            # Any remaining edge-case (shouldn’t happen after de-dup)
            return float('nan')

    def rbo_score(a_ids, b_ids, p):
    # you already have a safe version above; using it keeps behavior consistent
        return rbo_score_safe(a_ids, b_ids, p)

    def load_input_ids(root, size, mode, qid, inp_idx):
        """Read input_docids from run0 (identical to run1 for inputs)."""
        f = os.path.join(root, f"N{size}", mode, str(qid), f"in{inp_idx:02d}_run0.json")
        if not os.path.exists(f):
            return None
        try:
            return json.load(open(f))["input_docids"]
        except Exception:
            return None

    def load_output_ids(root, size, mode, qid, inp_idx, run):
        f = os.path.join(root, f"N{size}", mode, str(qid), f"in{inp_idx:02d}_run{run}.json")
        if not os.path.exists(f):
            return None
        try:
            return json.load(open(f))["output_docids"]
        except Exception:
            return None

    def kendall_tau_for_shared(a_list, b_list, shared_ids):
        """Compute Kendall’s τ restricted to shared docids."""
        from scipy.stats import kendalltau
        if not shared_ids:
            return np.nan
        a_filtered = [d for d in a_list if d in shared_ids]
        b_filtered = [d for d in b_list if d in shared_ids]
        n = len(a_filtered)
        if n < 2 or len(b_filtered) != n:
            return np.nan
        a_pos = {d: i for i, d in enumerate(a_filtered)}
        seq_a = [a_pos[d] for d in b_filtered if d in a_pos]
        seq_b = list(range(len(seq_a)))
        if len(seq_a) < 2:
            return np.nan
        tau, _ = kendalltau(seq_a, seq_b)
        return tau

    # ---------- main loop ----------
    for dataset in DATASETS:
        TOPICS_NAME = "beir-v1.0.0-trec-covid-test" if dataset == "trec-covid" else "beir-v1.0.0-nfcorpus-test"
        qrels = get_qrels(TOPICS_NAME)

        for model in MODELS:
            RESULTS_ROOT = f"results/{dataset}_{model}"
            if not os.path.isdir(RESULTS_ROOT):
                print(f"⚠️  Skipping {RESULTS_ROOT} (folder not found)")
                continue

            OUTPUT_FILE = os.path.join(RESULTS_ROOT, "results_summary_shared_ids.txt")
            print(f"\n=== Processing {dataset} × {model} (shared-by-IDs) ===")

            records = []
            size_dirs = sorted([d for d in os.listdir(RESULTS_ROOT) if d.startswith("N")],
                               key=lambda x: int(x[1:]))

            for size_dir in size_dirs:
                k = int(size_dir[1:])
                for mode in MODES:
                    mode_dir = os.path.join(RESULTS_ROOT, size_dir, mode)
                    if not os.path.isdir(mode_dir):
                        continue
                    try:
                        qids = sorted(os.listdir(mode_dir))
                    except Exception:
                        continue

                    for qid in tqdm(qids, desc=f"{dataset}_{model} {mode} N{k}"):
                        for inp_idx in range(NUM_INPUTS):
                            in5  = load_input_ids(RESULTS_ROOT, 5,  mode, qid, inp_idx) if os.path.isdir(os.path.join(RESULTS_ROOT,"N5", mode, str(qid)))  else None
                            in10 = load_input_ids(RESULTS_ROOT, 10, mode, qid, inp_idx) if os.path.isdir(os.path.join(RESULTS_ROOT,"N10",mode, str(qid))) else None
                            in20 = load_input_ids(RESULTS_ROOT, 20, mode, qid, inp_idx) if os.path.isdir(os.path.join(RESULTS_ROOT,"N20",mode, str(qid))) else None

                            shared5  = set(in5).intersection(in10, in20) if (in5 and in10 and in20) else set()
                            shared10 = set(in10).intersection(in20) if (in10 and in20) else set()
                            shared20 = set(in20) if in20 else set()

                            out0 = load_output_ids(RESULTS_ROOT, k, mode, qid, inp_idx, 0)
                            out1 = load_output_ids(RESULTS_ROOT, k, mode, qid, inp_idx, 1)
                            if not out0 or not out1:
                                continue

                            tau5  = kendall_tau_for_shared(out0, out1, shared5)  if shared5  else np.nan
                            rbo5  = rbo_score([d for d in out0 if d in shared5],
                                              [d for d in out1 if d in shared5], RBO_P[5]) if shared5 else np.nan

                            tau10 = kendall_tau_for_shared(out0, out1, shared10) if shared10 else np.nan
                            rbo10 = rbo_score([d for d in out0 if d in shared10],
                                              [d for d in out1 if d in shared10], RBO_P[10]) if shared10 else np.nan

                            tau20 = kendall_tau_for_shared(out0, out1, shared20) if shared20 else np.nan
                            rbo20 = rbo_score([d for d in out0 if d in shared20],
                                              [d for d in out1 if d in shared20], RBO_P[20]) if shared20 else np.nan

                            qrel_dict = get_qrel_dict_robust(qrels, qid)
                            # ---- NEW: DCG@k (raw) alongside existing nDCG@k ----
                            dcg0   = dcg_for_out(out0, qrel_dict, k)
                            dcg1   = dcg_for_out(out1, qrel_dict, k)
                            dcg_mean = float(np.mean([dcg0, dcg1]))

                            ndcg0  = ndcg_for_out(out0, qrel_dict, k)
                            ndcg1  = ndcg_for_out(out1, qrel_dict, k)
                            ndcg_mean = float(np.mean([ndcg0, ndcg1]))

                            records.append({
                                "mode": mode, "size": k,
                                "tau5": tau5, "rbo5": rbo5,
                                "tau10": tau10, "rbo10": rbo10,
                                "tau20": tau20, "rbo20": rbo20,
                                "dcg": dcg_mean,          # <-- NEW
                                "ndcg": ndcg_mean
                            })

            df = pd.DataFrame(records)
            if df.empty:
                print(f"⚠️  No records found for {dataset} × {model}. Skipping.")
                continue

            summary = (
                df.groupby(["mode", "size"])[["tau5","rbo5","tau10","rbo10","tau20","rbo20","dcg","ndcg"]]
                .mean()
                .reset_index()
                .sort_values(["mode","size"])
            )

            with open(OUTPUT_FILE, "w") as out:
                out.write("Aggregated τ & RBO over SHARED IDs (5/10/20) + DCG@k + nDCG@k\n")
                out.write("=" * 106 + "\n")
                out.write(f"Dataset: {dataset}\nModel: {model}\n\n")
                out.write(f"{'Mode':<18}{'Size':<6}{'τ@5':>8}{'RBO@5':>8}{'τ@10':>8}{'RBO@10':>8}{'τ@20':>8}{'RBO@20':>8}{'DCG@k':>10}{'nDCG@k':>10}\n")
                out.write("-" * 106 + "\n")
                for _, r in summary.iterrows():
                    out.write(
                        f"{r['mode']:<18} N{int(r['size']):<5}"
                        f"{(r['tau5']  if pd.notna(r['tau5'])  else float('nan')):>8.3f}"
                        f"{(r['rbo5']  if pd.notna(r['rbo5'])  else float('nan')):>8.3f}"
                        f"{(r['tau10'] if pd.notna(r['tau10']) else float('nan')):>8.3f}"
                        f"{(r['rbo10'] if pd.notna(r['rbo10']) else float('nan')):>8.3f}"
                        f"{(r['tau20'] if pd.notna(r['tau20']) else float('nan')):>8.3f}"
                        f"{(r['rbo20'] if pd.notna(r['rbo20']) else float('nan')):>8.3f}"
                        f"{r['dcg']:>10.3f}"
                        f"{r['ndcg']:>10.3f}\n"
                    )
            print(f"✅ Summary (shared-by-IDs) saved to {OUTPUT_FILE}")

# Run when needed: 
# compute_all_metrics_shared_by_ids()


############################################################################################################    
########################### v2 - generating the RBO, NDCG and Tau results ########################################
# this version has delta and RR and SDCG as well + significanc
############################################################################################################

import os, json, numpy as np, pandas as pd
from math import log2
from collections import OrderedDict
from scipy.stats import kendalltau, mannwhitneyu, ttest_ind
from rbo import RankingSimilarity
from tqdm import tqdm
from pyserini.search import get_qrels

def compute_all_metrics_shared_by_ids():
    DATASETS = ["trec-covid", "nfcorpus"]
    MODELS   = ["gpt-3.5", "gpt-4o-mini"]
    MODES    = ["half_relevant", "single_relevant", "single_nonrelevant"]
    SIZES    = [5, 10, 20]
    NUM_INPUTS = 50

    RBO_P = {5: 0.63, 10: 0.79, 20: 0.89}

    # all metrics we want to test significance across N5/N10/N20
    METRICS_FOR_SIG = [
        # first-table metrics
        "tau5", "rbo5", "tau10", "rbo10", "tau20", "rbo20", "dcg", "ndcg",
        # extra-table metrics
        "dcg_in", "dcg", "delta_dcg",
        "ndcg_in", "ndcg", "delta_ndcg",
        "rr_in", "rr_out", "delta_rr",
        "scaled_dcg_in", "scaled_dcg", "delta_scaled_dcg",
    ]

    # ---------- helpers ----------
    def dcg_at_k(rels, k):
        return sum(rel / log2(i + 2) for i, rel in enumerate(rels[:k]))

    def dcg_for_out(out_docs, qrel_dict, k):
        """Raw DCG@k (no normalization, graded relevance)."""
        if not out_docs:
            return 0.0
        labels = [int(qrel_dict.get(d, 0)) for d in out_docs[:k]]
        return dcg_at_k(labels, k)

    def ndcg_for_out(out_docs, qrel_dict, k):
        """nDCG@k using graded relevance."""
        if not out_docs:
            return 0.0
        labels = [int(qrel_dict.get(d, 0)) for d in out_docs[:k]]
        dcg = dcg_at_k(labels, k)
        ideal = sorted(labels, reverse=True)
        idcg = dcg_at_k(ideal, k)
        return dcg / idcg if idcg > 0 else 0.0

    def reciprocal_rank(out_docs, qrel_dict):
        """RR = 1/rank_of_first_relevant; 0 if no relevant."""
        if not out_docs:
            return 0.0
        for i, d in enumerate(out_docs):
            if int(qrel_dict.get(d, 0)) > 0:
                return 1.0 / (i + 1)
        return 0.0

    def scaled_dcg_for_out(out_docs, qrel_dict, k):
        """
        Scaled DCG@k with binary relevance:
        - Numerator: DCG with rel = 1 if qrel > 0 else 0
        - Denominator: DCG if *all* docs in the list (up to k) had rel = 1
        """
        if not out_docs:
            return 0.0
        labels_bin = [1 if int(qrel_dict.get(d, 0)) > 0 else 0 for d in out_docs[:k]]
        num_dcg = dcg_at_k(labels_bin, k)
        all_ones = [1] * len(labels_bin)
        denom_dcg = dcg_at_k(all_ones, k)
        return num_dcg / denom_dcg if denom_dcg > 0 else 0.0

    def rbo_score_safe(a_ids, b_ids, p):
        a = list(OrderedDict.fromkeys(a_ids or []))
        b = list(OrderedDict.fromkeys(b_ids or []))
        if len(a) < 2 or len(b) < 2:
            return float('nan')
        try:
            return RankingSimilarity(a, b).rbo(p=p)
        except AssertionError:
            return float('nan')

    def rbo_score(a_ids, b_ids, p):
        return rbo_score_safe(a_ids, b_ids, p)

    def load_input_ids(root, size, mode, qid, inp_idx):
        f = os.path.join(root, f"N{size}", mode, str(qid), f"in{inp_idx:02d}_run0.json")
        if not os.path.exists(f):
            return None
        try:
            return json.load(open(f))["input_docids"]
        except Exception:
            return None

    def load_output_ids(root, size, mode, qid, inp_idx, run):
        f = os.path.join(root, f"N{size}", mode, str(qid), f"in{inp_idx:02d}_run{run}.json")
        if not os.path.exists(f):
            return None
        try:
            return json.load(open(f))["output_docids"]
        except Exception:
            return None

    def kendall_tau_for_shared(a_list, b_list, shared_ids):
        if not shared_ids:
            return np.nan
        a_filtered = [d for d in a_list if d in shared_ids]
        b_filtered = [d for d in b_list if d in shared_ids]
        n = len(a_filtered)
        if n < 2 or len(b_filtered) != n:
            return np.nan
        a_pos = {d: i for i, d in enumerate(a_filtered)}
        seq_a = [a_pos[d] for d in b_filtered if d in a_pos]
        seq_b = list(range(len(seq_a)))
        if len(seq_a) < 2:
            return np.nan
        tau, _ = kendalltau(seq_a, seq_b)
        return tau

    def significant(a, b):
        """
        Significance decision between two samples a, b using:
        - Mann–Whitney U
        - Welch's t-test
        Returns True if both tests have p < 0.05.
        """
        a = np.asarray(a)
        b = np.asarray(b)
        if len(a) < 2 or len(b) < 2:
            return False, np.nan, np.nan
        p_u = mannwhitneyu(a, b, alternative="two-sided").pvalue
        p_t = ttest_ind(a, b, equal_var=False).pvalue
        return (p_u < 0.05 and p_t < 0.05), p_u, p_t

    def compute_sig_marks_and_lines(df, dataset, model):
        """
        For each mode & metric, compare the three sizes N5/N10/N20.
        Returns:
          marks[metric][(mode, size)] -> mark string ("", "*", "†", "**")
          lines -> list of text lines summarising significance decisions.
        """
        marks = {m: {} for m in METRICS_FOR_SIG}
        lines = []

        for mode in sorted(df["mode"].unique()):
            df_mode = df[df["mode"] == mode]
            for metric in METRICS_FOR_SIG:
                vals = {}
                means = {}
                for s in SIZES:
                    arr = df_mode[df_mode["size"] == s][metric].dropna().to_numpy()
                    if arr.size > 0:
                        vals[s] = arr
                        means[s] = float(np.mean(arr))

                if len(vals) < 3:
                    continue

                ordered = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
                best_size, best_mean = ordered[0]
                mid_size, mid_mean   = ordered[1]
                low_size, low_mean   = ordered[2]

                sig_bm, p_u_bm, p_t_bm = significant(vals[best_size], vals[mid_size])
                sig_bl, p_u_bl, p_t_bl = significant(vals[best_size], vals[low_size])
                sig_ml, p_u_ml, p_t_ml = significant(vals[mid_size],  vals[low_size])

                mark_best = ""
                if sig_bm and sig_bl:
                    mark_best = "**"
                elif sig_bm and not sig_bl:
                    mark_best = "*"
                elif (not sig_bm) and sig_bl:
                    mark_best = "†"

                if mark_best:
                    marks[metric][(mode, best_size)] = mark_best

                if sig_ml:
                    prev = marks[metric].get((mode, mid_size), "")
                    marks[metric][(mode, mid_size)] = prev + "*"

                line = (
                    f"{dataset} | {model} | mode={mode} | metric={metric} | "
                    f"best=N{best_size} (mean={best_mean:.4f}), "
                    f"mid=N{mid_size} (mean={mid_mean:.4f}), "
                    f"low=N{low_size} (mean={low_mean:.4f}) | "
                    f"sig(best>mid)={sig_bm}, sig(best>low)={sig_bl}, sig(mid>low)={sig_ml}"
                )
                lines.append(line)

        return marks, lines

    # ---------- main loop ----------
    for dataset in DATASETS:
        TOPICS_NAME = "beir-v1.0.0-trec-covid-test" if dataset == "trec-covid" else "beir-v1.0.0-nfcorpus-test"
        qrels = get_qrels(TOPICS_NAME)

        for model in MODELS:
            RESULTS_ROOT = f"results/{dataset}_{model}"
            if not os.path.isdir(RESULTS_ROOT):
                print(f"⚠️  Skipping {RESULTS_ROOT} (folder not found)")
                continue

            OUTPUT_FILE = os.path.join(RESULTS_ROOT, "results_summary_ids.txt")
            print(f"\n=== Processing {dataset} × {model} (shared-by-IDs) ===")

            records = []
            size_dirs = sorted(
                [d for d in os.listdir(RESULTS_ROOT) if d.startswith("N")],
                key=lambda x: int(x[1:])
            )

            for size_dir in size_dirs:
                k = int(size_dir[1:])
                for mode in MODES:
                    mode_dir = os.path.join(RESULTS_ROOT, size_dir, mode)
                    if not os.path.isdir(mode_dir):
                        continue
                    try:
                        qids = sorted(os.listdir(mode_dir))
                    except Exception:
                        continue

                    for qid in tqdm(qids, desc=f"{dataset}_{model} {mode} N{k}"):
                        for inp_idx in range(NUM_INPUTS):
                            in5  = load_input_ids(RESULTS_ROOT, 5,  mode, qid, inp_idx) if os.path.isdir(os.path.join(RESULTS_ROOT,"N5", mode, str(qid)))  else None
                            in10 = load_input_ids(RESULTS_ROOT, 10, mode, qid, inp_idx) if os.path.isdir(os.path.join(RESULTS_ROOT,"N10",mode, str(qid))) else None
                            in20 = load_input_ids(RESULTS_ROOT, 20, mode, qid, inp_idx) if os.path.isdir(os.path.join(RESULTS_ROOT,"N20",mode, str(qid))) else None

                            shared5  = set(in5).intersection(in10, in20) if (in5 and in10 and in20) else set()
                            shared10 = set(in10).intersection(in20)       if (in10 and in20)       else set()
                            shared20 = set(in20) if in20 else set()

                            out0 = load_output_ids(RESULTS_ROOT, k, mode, qid, inp_idx, 0)
                            out1 = load_output_ids(RESULTS_ROOT, k, mode, qid, inp_idx, 1)
                            if not out0 or not out1:
                                continue

                            tau5  = kendall_tau_for_shared(out0, out1, shared5)  if shared5  else np.nan
                            rbo5  = rbo_score([d for d in out0 if d in shared5],
                                              [d for d in out1 if d in shared5], RBO_P[5])  if shared5  else np.nan

                            tau10 = kendall_tau_for_shared(out0, out1, shared10) if shared10 else np.nan
                            rbo10 = rbo_score([d for d in out0 if d in shared10],
                                              [d for d in out1 if d in shared10], RBO_P[10]) if shared10 else np.nan

                            tau20 = kendall_tau_for_shared(out0, out1, shared20) if shared20 else np.nan
                            rbo20 = rbo_score([d for d in out0 if d in shared20],
                                              [d for d in out1 if d in shared20], RBO_P[20]) if shared20 else np.nan

                            qrel_dict = get_qrel_dict_robust(qrels, qid)

                            dcg0   = dcg_for_out(out0, qrel_dict, k)
                            dcg1   = dcg_for_out(out1, qrel_dict, k)
                            dcg_mean = float(np.mean([dcg0, dcg1]))

                            ndcg0  = ndcg_for_out(out0, qrel_dict, k)
                            ndcg1  = ndcg_for_out(out1, qrel_dict, k)
                            ndcg_mean = float(np.mean([ndcg0, ndcg1]))

                            rr0 = reciprocal_rank(out0, qrel_dict)
                            rr1 = reciprocal_rank(out1, qrel_dict)
                            rr_out_mean = float(np.mean([rr0, rr1]))

                            scaled_dcg0 = scaled_dcg_for_out(out0, qrel_dict, k)
                            scaled_dcg1 = scaled_dcg_for_out(out1, qrel_dict, k)
                            scaled_dcg_mean = float(np.mean([scaled_dcg0, scaled_dcg1]))

                            if k == 5:
                                inp_docs = in5
                            elif k == 10:
                                inp_docs = in10
                            elif k == 20:
                                inp_docs = in20
                            else:
                                inp_docs = None

                            if inp_docs:
                                dcg_in = dcg_for_out(inp_docs, qrel_dict, k)
                                ndcg_in = ndcg_for_out(inp_docs, qrel_dict, k)
                                rr_in = reciprocal_rank(inp_docs, qrel_dict)
                                scaled_dcg_in = scaled_dcg_for_out(inp_docs, qrel_dict, k)

                                delta_dcg = dcg_mean - dcg_in
                                delta_ndcg = ndcg_mean - ndcg_in
                                delta_rr = rr_out_mean - rr_in
                                delta_scaled_dcg = scaled_dcg_mean - scaled_dcg_in
                            else:
                                dcg_in = ndcg_in = rr_in = scaled_dcg_in = np.nan
                                delta_dcg = delta_ndcg = delta_rr = delta_scaled_dcg = np.nan

                            records.append({
                                "mode": mode,
                                "size": k,
                                "tau5": tau5, "rbo5": rbo5,
                                "tau10": tau10, "rbo10": rbo10,
                                "tau20": tau20, "rbo20": rbo20,
                                "dcg": dcg_mean,
                                "ndcg": ndcg_mean,
                                "dcg_in": dcg_in,
                                "ndcg_in": ndcg_in,
                                "delta_dcg": delta_dcg,
                                "delta_ndcg": delta_ndcg,
                                "rr_in": rr_in,
                                "rr_out": rr_out_mean,
                                "delta_rr": delta_rr,
                                "scaled_dcg_in": scaled_dcg_in,
                                "scaled_dcg": scaled_dcg_mean,
                                "delta_scaled_dcg": delta_scaled_dcg,
                            })

            df = pd.DataFrame(records)
            if df.empty:
                print(f"⚠️  No records found for {dataset} × {model}. Skipping.")
                continue

            # significance marks (for both tables)
            sig_marks, sig_lines = compute_sig_marks_and_lines(df, dataset, model)

            def fmt_basic(metric, mode, size, value, width):
                mark = sig_marks.get(metric, {}).get((mode, size), "")
                if pd.isna(value):
                    v = "nan"
                else:
                    v = f"{value:.3f}"
                return f"{v}{mark}".rjust(width)

            def fmt_extra(metric, mode, size, value, width):
                mark = sig_marks.get(metric, {}).get((mode, size), "")
                if pd.isna(value):
                    v = "nan"
                else:
                    v = f"{value:.3f}"
                return f"{v}{mark}".rjust(width)

            # ---------- FIRST TABLE ----------
            summary_basic = (
                df.groupby(["mode", "size"])[
                    ["tau5","rbo5","tau10","rbo10","tau20","rbo20","dcg","ndcg"]
                ]
                .mean()
                .reset_index()
                .sort_values(["mode","size"])
            )

            with open(OUTPUT_FILE, "w") as out:
                out.write("Aggregated τ & RBO over SHARED IDs (5/10/20) + DCG@k + nDCG@k\n")
                out.write("=" * 106 + "\n")
                out.write(f"Dataset: {dataset}\nModel: {model}\n\n")
                out.write(
                    f"{'Mode':<18}{'Size':<6}"
                    f"{'τ@5':>8}{'RBO@5':>8}"
                    f"{'τ@10':>8}{'RBO@10':>8}"
                    f"{'τ@20':>8}{'RBO@20':>8}"
                    f"{'DCG@k':>10}{'nDCG@k':>10}\n"
                )
                out.write("-" * 106 + "\n")
                for _, r in summary_basic.iterrows():
                    mode = r["mode"]
                    size = int(r["size"])
                    out.write(
                        f"{mode:<18} N{size:<5}"
                        f"{fmt_basic('tau5',  mode, size, r['tau5'],  8)}"
                        f"{fmt_basic('rbo5',  mode, size, r['rbo5'],  8)}"
                        f"{fmt_basic('tau10', mode, size, r['tau10'], 8)}"
                        f"{fmt_basic('rbo10', mode, size, r['rbo10'], 8)}"
                        f"{fmt_basic('tau20', mode, size, r['tau20'], 8)}"
                        f"{fmt_basic('rbo20', mode, size, r['rbo20'], 8)}"
                        f"{fmt_basic('dcg',   mode, size, r['dcg'],  10)}"
                        f"{fmt_basic('ndcg',  mode, size, r['ndcg'], 10)}\n"
                    )

            # ---------- EXTRA TABLE ----------
            summary_extra = (
                df.groupby(["mode", "size"])[
                    [
                        "dcg_in", "dcg", "delta_dcg",
                        "ndcg_in", "ndcg", "delta_ndcg",
                        "rr_in", "rr_out", "delta_rr",
                        "scaled_dcg_in",
                        "scaled_dcg",          # ⬅️ ADD THIS
                        "delta_scaled_dcg",
                    ]
                ]
                .mean()
                .reset_index()
                .sort_values(["mode", "size"])
            )


            with open(OUTPUT_FILE, "a") as out:
                out.write("\n\nAdditional metrics vs INPUT ranking (averaged over runs)\n")
                out.write("=" * 160 + "\n")
                out.write(
                    f"{'Mode':<18}{'Size':<6}"
                    f"{'DCG_in':>12}{'DCG_out':>12}{'ΔDCG':>12}"
                    f"{'nDCG_in':>12}{'nDCG_out':>12}{'ΔnDCG':>12}"
                    f"{'RR_in':>12}{'RR_out':>12}{'ΔRR':>12}"
                    f"{'sDCG_in':>12}{'sDCG_out':>12}{'ΔsDCG':>12}\n"
                )
                out.write("-" * 160 + "\n")
                for _, r in summary_extra.iterrows():
                    mode = r["mode"]
                    size = int(r["size"])
                    out.write(
                        f"{mode:<18} N{size:<5}"
                        f"{fmt_extra('dcg_in',          mode, size, r['dcg_in'],          12)}"
                        f"{fmt_extra('dcg',             mode, size, r['dcg'],             12)}"
                        f"{fmt_extra('delta_dcg',       mode, size, r['delta_dcg'],       12)}"
                        f"{fmt_extra('ndcg_in',         mode, size, r['ndcg_in'],         12)}"
                        f"{fmt_extra('ndcg',            mode, size, r['ndcg'],            12)}"
                        f"{fmt_extra('delta_ndcg',      mode, size, r['delta_ndcg'],      12)}"
                        f"{fmt_extra('rr_in',           mode, size, r['rr_in'],           12)}"
                        f"{fmt_extra('rr_out',          mode, size, r['rr_out'],          12)}"
                        f"{fmt_extra('delta_rr',        mode, size, r['delta_rr'],        12)}"
                        f"{fmt_extra('scaled_dcg_in',   mode, size, r['scaled_dcg_in'],   12)}"
                        f"{fmt_extra('scaled_dcg',      mode, size, r['scaled_dcg'],      12)}"
                        f"{fmt_extra('delta_scaled_dcg',mode, size, r['delta_scaled_dcg'],12)}\n"
                    )

                out.write("\nSignificance across sizes (N5, N10, N20) per mode & metric\n")
                out.write("Marks: ** best>mid & best>low; * best>mid only or mid>low; † best>low only\n")
                for line in sig_lines:
                    out.write(line + "\n")

            print(f"✅ Summary (shared-by-IDs) saved to {OUTPUT_FILE}")



# Run when needed: 
# compute_all_metrics_shared_by_ids()



############################################################################################################################
####################### V3 - same as above with all metrics and significance but this time for shared Docs #######################
####################### here we dont have ERR and for SDCG we consider the all rel=1 and dcg as binarized #################
#############################################################################################################################

# import os, json, numpy as np, pandas as pd
# from math import log2
# from collections import OrderedDict
# from scipy.stats import kendalltau, mannwhitneyu, ttest_ind
# from rbo import RankingSimilarity
# from tqdm import tqdm
# from pyserini.search import get_qrels

# def compute_all_metrics_shared_by_ids():
#     DATASETS = ["trec-covid", "nfcorpus"]
#     MODELS   = ["gpt-3.5", "gpt-4o-mini"]
#     MODES    = ["half_relevant", "single_relevant", "single_nonrelevant"]
#     SIZES    = [5, 10, 20]
#     NUM_INPUTS = 50

#     RBO_P = {5: 0.63, 10: 0.79, 20: 0.89}

#     # All metrics we test significance across N5/N10/N20
#     METRICS_FOR_SIG = [
#         # τ/RBO on shared sets
#         "tau5", "rbo5", "tau10", "rbo10", "tau20", "rbo20",

#         # full-list metrics (existing behaviour)
#         "dcg_in", "dcg", "delta_dcg",
#         "ndcg_in", "ndcg", "delta_ndcg",
#         "rr_in", "rr_out", "delta_rr",
#         "scaled_dcg_in", "scaled_dcg", "delta_scaled_dcg",

#         # NEW: metrics on shared-5
#         "dcg5_in", "dcg5", "delta_dcg5",
#         "ndcg5_in", "ndcg5", "delta_ndcg5",
#         "rr5_in", "rr5_out", "delta_rr5",
#         "scaled_dcg5_in", "scaled_dcg5", "delta_scaled_dcg5",

#         # NEW: metrics on shared-10
#         "dcg10_in", "dcg10", "delta_dcg10",
#         "ndcg10_in", "ndcg10", "delta_ndcg10",
#         "rr10_in", "rr10_out", "delta_rr10",
#         "scaled_dcg10_in", "scaled_dcg10", "delta_scaled_dcg10",
#     ]

#     # ---------- helpers ----------
#     def dcg_at_k(rels, k):
#         return sum(rel / log2(i + 2) for i, rel in enumerate(rels[:k]))

#     def dcg_for_out(out_docs, qrel_dict, k):
#         """Raw DCG@k (no normalization, graded relevance)."""
#         if not out_docs:
#             return 0.0
#         labels = [int(qrel_dict.get(d, 0)) for d in out_docs[:k]]
#         return dcg_at_k(labels, k)

#     def ndcg_for_out(out_docs, qrel_dict, k):
#         """nDCG@k using graded relevance."""
#         if not out_docs:
#             return 0.0
#         labels = [int(qrel_dict.get(d, 0)) for d in out_docs[:k]]
#         dcg = dcg_at_k(labels, k)
#         ideal = sorted(labels, reverse=True)
#         idcg = dcg_at_k(ideal, k)
#         return dcg / idcg if idcg > 0 else 0.0

#     def reciprocal_rank(out_docs, qrel_dict):
#         """RR = 1/rank_of_first_relevant; 0 if no relevant."""
#         if not out_docs:
#             return 0.0
#         for i, d in enumerate(out_docs):
#             if int(qrel_dict.get(d, 0)) > 0:
#                 return 1.0 / (i + 1)
#         return 0.0

#     def scaled_dcg_for_out(out_docs, qrel_dict, k):
#         """
#         Scaled DCG@k with binary relevance:
#         - Numerator: DCG with rel = 1 if qrel > 0 else 0
#         - Denominator: DCG if *all* docs in the list (up to k) had rel = 1
#         """
#         if not out_docs:
#             return 0.0
#         labels_bin = [1 if int(qrel_dict.get(d, 0)) > 0 else 0 for d in out_docs[:k]]
#         num_dcg = dcg_at_k(labels_bin, k)
#         all_ones = [1] * len(labels_bin)
#         denom_dcg = dcg_at_k(all_ones, k)
#         return num_dcg / denom_dcg if denom_dcg > 0 else 0.0

#     def rbo_score_safe(a_ids, b_ids, p):
#         a = list(OrderedDict.fromkeys(a_ids or []))
#         b = list(OrderedDict.fromkeys(b_ids or []))
#         if len(a) < 2 or len(b) < 2:
#             return float('nan')
#         try:
#             return RankingSimilarity(a, b).rbo(p=p)
#         except AssertionError:
#             return float('nan')

#     def rbo_score(a_ids, b_ids, p):
#         return rbo_score_safe(a_ids, b_ids, p)

#     def load_input_ids(root, size, mode, qid, inp_idx):
#         f = os.path.join(root, f"N{size}", mode, str(qid), f"in{inp_idx:02d}_run0.json")
#         if not os.path.exists(f):
#             return None
#         try:
#             return json.load(open(f))["input_docids"]
#         except Exception:
#             return None

#     def load_output_ids(root, size, mode, qid, inp_idx, run):
#         f = os.path.join(root, f"N{size}", mode, str(qid), f"in{inp_idx:02d}_run{run}.json")
#         if not os.path.exists(f):
#             return None
#         try:
#             return json.load(open(f))["output_docids"]
#         except Exception:
#             return None

#     def kendall_tau_for_shared(a_list, b_list, shared_ids):
#         if not shared_ids:
#             return np.nan
#         a_filtered = [d for d in a_list if d in shared_ids]
#         b_filtered = [d for d in b_list if d in shared_ids]
#         n = len(a_filtered)
#         if n < 2 or len(b_filtered) != n:
#             return np.nan
#         a_pos = {d: i for i, d in enumerate(a_filtered)}
#         seq_a = [a_pos[d] for d in b_filtered if d in a_pos]
#         seq_b = list(range(len(seq_a)))
#         if len(seq_a) < 2:
#             return np.nan
#         tau, _ = kendalltau(seq_a, seq_b)
#         return tau

#     def significant(a, b):
#         """
#         Significance decision between two samples a, b using:
#         - Mann–Whitney U
#         - Welch's t-test
#         Returns True if both tests have p < 0.05.
#         """
#         a = np.asarray(a)
#         b = np.asarray(b)
#         if len(a) < 2 or len(b) < 2:
#             return False, np.nan, np.nan
#         p_u = mannwhitneyu(a, b, alternative="two-sided").pvalue
#         p_t = ttest_ind(a, b, equal_var=False).pvalue
#         return (p_u < 0.05 and p_t < 0.05), p_u, p_t

#     def compute_sig_marks_and_lines(df, dataset, model):
#         """
#         For each mode & metric, compare the three sizes N5/N10/N20.
#         Returns:
#           marks[metric][(mode, size)] -> mark string ("", "*", "†", "**")
#           lines -> list of text lines summarising significance decisions.
#         """
#         marks = {m: {} for m in METRICS_FOR_SIG}
#         lines = []

#         for mode in sorted(df["mode"].unique()):
#             df_mode = df[df["mode"] == mode]
#             for metric in METRICS_FOR_SIG:
#                 if metric not in df_mode.columns:
#                     continue
#                 vals = {}
#                 means = {}
#                 for s in SIZES:
#                     arr = df_mode[df_mode["size"] == s][metric].dropna().to_numpy()
#                     if arr.size > 0:
#                         vals[s] = arr
#                         means[s] = float(np.mean(arr))

#                 if len(vals) < 3:
#                     continue

#                 ordered = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
#                 best_size, best_mean = ordered[0]
#                 mid_size, mid_mean   = ordered[1]
#                 low_size, low_mean   = ordered[2]

#                 sig_bm, p_u_bm, p_t_bm = significant(vals[best_size], vals[mid_size])
#                 sig_bl, p_u_bl, p_t_bl = significant(vals[best_size], vals[low_size])
#                 sig_ml, p_u_ml, p_t_ml = significant(vals[mid_size],  vals[low_size])

#                 mark_best = ""
#                 if sig_bm and sig_bl:
#                     mark_best = "**"
#                 elif sig_bm and not sig_bl:
#                     mark_best = "*"
#                 elif (not sig_bm) and sig_bl:
#                     mark_best = "†"

#                 if mark_best:
#                     marks[metric][(mode, best_size)] = mark_best

#                 if sig_ml:
#                     prev = marks[metric].get((mode, mid_size), "")
#                     marks[metric][(mode, mid_size)] = prev + "*"

#                 line = (
#                     f"{dataset} | {model} | mode={mode} | metric={metric} | "
#                     f"best=N{best_size} (mean={best_mean:.4f}), "
#                     f"mid=N{mid_size} (mean={mid_mean:.4f}), "
#                     f"low=N{low_size} (mean={low_mean:.4f}) | "
#                     f"sig(best>mid)={sig_bm}, sig(best>low)={sig_bl}, sig(mid>low)={sig_ml}"
#                 )
#                 lines.append(line)

#         return marks, lines

#     def compute_shared_metrics(doc_list, shared_ids, qrel_dict, k_target):
#         """
#         Restrict doc_list to shared_ids (preserve order) and compute
#         DCG@k_target, nDCG@k_target, RR, scaled DCG.
#         Returns (dcg, ndcg, rr, sdcg). NaNs if no shared docs.
#         """
#         if not doc_list or not shared_ids:
#             return np.nan, np.nan, np.nan, np.nan
#         restricted = [d for d in doc_list if d in shared_ids]
#         if len(restricted) == 0:
#             return np.nan, np.nan, np.nan, np.nan
#         dcg = dcg_for_out(restricted, qrel_dict, k_target)
#         ndcg = ndcg_for_out(restricted, qrel_dict, k_target)
#         rr = reciprocal_rank(restricted, qrel_dict)
#         sdcg = scaled_dcg_for_out(restricted, qrel_dict, k_target)
#         return dcg, ndcg, rr, sdcg

#     # ---------- main loop ----------
#     for dataset in DATASETS:
#         TOPICS_NAME = "beir-v1.0.0-trec-covid-test" if dataset == "trec-covid" else "beir-v1.0.0-nfcorpus-test"
#         qrels = get_qrels(TOPICS_NAME)

#         for model in MODELS:
#             RESULTS_ROOT = f"results/{dataset}_{model}"
#             if not os.path.isdir(RESULTS_ROOT):
#                 print(f"⚠️  Skipping {RESULTS_ROOT} (folder not found)")
#                 continue

#             OUTPUT_FILE = os.path.join(RESULTS_ROOT, "results_summary_shared_ids.txt")
#             print(f"\n=== Processing {dataset} × {model} (shared-by-IDs) ===")

#             records = []
#             size_dirs = sorted(
#                 [d for d in os.listdir(RESULTS_ROOT) if d.startswith("N")],
#                 key=lambda x: int(x[1:])
#             )

#             for size_dir in size_dirs:
#                 k = int(size_dir[1:])
#                 for mode in MODES:
#                     mode_dir = os.path.join(RESULTS_ROOT, size_dir, mode)
#                     if not os.path.isdir(mode_dir):
#                         continue
#                     try:
#                         qids = sorted(os.listdir(mode_dir))
#                     except Exception:
#                         continue

#                     for qid in tqdm(qids, desc=f"{dataset}_{model} {mode} N{k}"):
#                         for inp_idx in range(NUM_INPUTS):
#                             in5  = load_input_ids(RESULTS_ROOT, 5,  mode, qid, inp_idx) if os.path.isdir(os.path.join(RESULTS_ROOT,"N5", mode, str(qid)))  else None
#                             in10 = load_input_ids(RESULTS_ROOT, 10, mode, qid, inp_idx) if os.path.isdir(os.path.join(RESULTS_ROOT,"N10",mode, str(qid))) else None
#                             in20 = load_input_ids(RESULTS_ROOT, 20, mode, qid, inp_idx) if os.path.isdir(os.path.join(RESULTS_ROOT,"N20",mode, str(qid))) else None

#                             shared5  = set(in5).intersection(in10, in20) if (in5 and in10 and in20) else set()
#                             shared10 = set(in10).intersection(in20)       if (in10 and in20)       else set()
#                             shared20 = set(in20) if in20 else set()

#                             out0 = load_output_ids(RESULTS_ROOT, k, mode, qid, inp_idx, 0)
#                             out1 = load_output_ids(RESULTS_ROOT, k, mode, qid, inp_idx, 1)
#                             if not out0 or not out1:
#                                 continue

#                             # τ / RBO on shared sets
#                             tau5  = kendall_tau_for_shared(out0, out1, shared5)  if shared5  else np.nan
#                             rbo5  = rbo_score([d for d in out0 if d in shared5],
#                                               [d for d in out1 if d in shared5], RBO_P[5])  if shared5  else np.nan

#                             tau10 = kendall_tau_for_shared(out0, out1, shared10) if shared10 else np.nan
#                             rbo10 = rbo_score([d for d in out0 if d in shared10],
#                                               [d for d in out1 if d in shared10], RBO_P[10]) if shared10 else np.nan

#                             tau20 = kendall_tau_for_shared(out0, out1, shared20) if shared20 else np.nan
#                             rbo20 = rbo_score([d for d in out0 if d in shared20],
#                                               [d for d in out1 if d in shared20], RBO_P[20]) if shared20 else np.nan

#                             # qrels
#                             qrel_dict = get_qrel_dict_robust(qrels, qid)

#                             # full-list output metrics @k (existing)
#                             dcg0   = dcg_for_out(out0, qrel_dict, k)
#                             dcg1   = dcg_for_out(out1, qrel_dict, k)
#                             dcg_mean = float(np.mean([dcg0, dcg1]))

#                             ndcg0  = ndcg_for_out(out0, qrel_dict, k)
#                             ndcg1  = ndcg_for_out(out1, qrel_dict, k)
#                             ndcg_mean = float(np.mean([ndcg0, ndcg1]))

#                             rr0 = reciprocal_rank(out0, qrel_dict)
#                             rr1 = reciprocal_rank(out1, qrel_dict)
#                             rr_out_mean = float(np.mean([rr0, rr1]))

#                             scaled_dcg0 = scaled_dcg_for_out(out0, qrel_dict, k)
#                             scaled_dcg1 = scaled_dcg_for_out(out1, qrel_dict, k)
#                             scaled_dcg_mean = float(np.mean([scaled_dcg0, scaled_dcg1]))

#                             # choose input docs for this size
#                             if k == 5:
#                                 inp_docs = in5
#                             elif k == 10:
#                                 inp_docs = in10
#                             elif k == 20:
#                                 inp_docs = in20
#                             else:
#                                 inp_docs = None

#                             # base metrics vs input (full list)
#                             if inp_docs:
#                                 dcg_in = dcg_for_out(inp_docs, qrel_dict, k)
#                                 ndcg_in = ndcg_for_out(inp_docs, qrel_dict, k)
#                                 rr_in = reciprocal_rank(inp_docs, qrel_dict)
#                                 scaled_dcg_in = scaled_dcg_for_out(inp_docs, qrel_dict, k)

#                                 delta_dcg = dcg_mean - dcg_in
#                                 delta_ndcg = ndcg_mean - ndcg_in
#                                 delta_rr = rr_out_mean - rr_in
#                                 delta_scaled_dcg = scaled_dcg_mean - scaled_dcg_in
#                             else:
#                                 dcg_in = ndcg_in = rr_in = scaled_dcg_in = np.nan
#                                 delta_dcg = delta_ndcg = delta_rr = delta_scaled_dcg = np.nan

#                             # ===== NEW: metrics on shared-5 & shared-10 =====
#                             # shared-5 metrics (if shared5 non-empty and we have input)
#                             if inp_docs and shared5:
#                                 dcg5_in, ndcg5_in, rr5_in, sdcg5_in = compute_shared_metrics(
#                                     inp_docs, shared5, qrel_dict, 5
#                                 )
#                                 dcg5_0, ndcg5_0, rr5_0, sdcg5_0 = compute_shared_metrics(
#                                     out0, shared5, qrel_dict, 5
#                                 )
#                                 dcg5_1, ndcg5_1, rr5_1, sdcg5_1 = compute_shared_metrics(
#                                     out1, shared5, qrel_dict, 5
#                                 )
#                                 dcg5 = float(np.nanmean([dcg5_0, dcg5_1]))
#                                 ndcg5 = float(np.nanmean([ndcg5_0, ndcg5_1]))
#                                 rr5_out = float(np.nanmean([rr5_0, rr5_1]))
#                                 sdcg5 = float(np.nanmean([sdcg5_0, sdcg5_1]))

#                                 delta_dcg5 = dcg5 - dcg5_in
#                                 delta_ndcg5 = ndcg5 - ndcg5_in
#                                 delta_rr5 = rr5_out - rr5_in
#                                 delta_scaled_dcg5 = sdcg5 - sdcg5_in
#                             else:
#                                 dcg5_in = ndcg5_in = rr5_in = sdcg5_in = np.nan
#                                 dcg5 = ndcg5 = rr5_out = sdcg5 = np.nan
#                                 delta_dcg5 = delta_ndcg5 = delta_rr5 = delta_scaled_dcg5 = np.nan

#                             # shared-10 metrics (if shared10 non-empty and we have input)
#                             if inp_docs and shared10:
#                                 dcg10_in, ndcg10_in, rr10_in, sdcg10_in = compute_shared_metrics(
#                                     inp_docs, shared10, qrel_dict, 10
#                                 )
#                                 dcg10_0, ndcg10_0, rr10_0, sdcg10_0 = compute_shared_metrics(
#                                     out0, shared10, qrel_dict, 10
#                                 )
#                                 dcg10_1, ndcg10_1, rr10_1, sdcg10_1 = compute_shared_metrics(
#                                     out1, shared10, qrel_dict, 10
#                                 )
#                                 dcg10 = float(np.nanmean([dcg10_0, dcg10_1]))
#                                 ndcg10 = float(np.nanmean([ndcg10_0, ndcg10_1]))
#                                 rr10_out = float(np.nanmean([rr10_0, rr10_1]))
#                                 sdcg10 = float(np.nanmean([sdcg10_0, sdcg10_1]))

#                                 delta_dcg10 = dcg10 - dcg10_in
#                                 delta_ndcg10 = ndcg10 - ndcg10_in
#                                 delta_rr10 = rr10_out - rr10_in
#                                 delta_scaled_dcg10 = sdcg10 - sdcg10_in
#                             else:
#                                 dcg10_in = ndcg10_in = rr10_in = sdcg10_in = np.nan
#                                 dcg10 = ndcg10 = rr10_out = sdcg10 = np.nan
#                                 delta_dcg10 = delta_ndcg10 = delta_rr10 = delta_scaled_dcg10 = np.nan

#                             records.append({
#                                 "mode": mode,
#                                 "size": k,

#                                 "tau5": tau5, "rbo5": rbo5,
#                                 "tau10": tau10, "rbo10": rbo10,
#                                 "tau20": tau20, "rbo20": rbo20,

#                                 # full-list metrics
#                                 "dcg": dcg_mean,
#                                 "ndcg": ndcg_mean,
#                                 "dcg_in": dcg_in,
#                                 "ndcg_in": ndcg_in,
#                                 "delta_dcg": delta_dcg,
#                                 "delta_ndcg": delta_ndcg,
#                                 "rr_in": rr_in,
#                                 "rr_out": rr_out_mean,
#                                 "delta_rr": delta_rr,
#                                 "scaled_dcg_in": scaled_dcg_in,
#                                 "scaled_dcg": scaled_dcg_mean,
#                                 "delta_scaled_dcg": delta_scaled_dcg,

#                                 # shared-5 metrics
#                                 "dcg5_in": dcg5_in,
#                                 "dcg5": dcg5,
#                                 "delta_dcg5": delta_dcg5,
#                                 "ndcg5_in": ndcg5_in,
#                                 "ndcg5": ndcg5,
#                                 "delta_ndcg5": delta_ndcg5,
#                                 "rr5_in": rr5_in,
#                                 "rr5_out": rr5_out,
#                                 "delta_rr5": delta_rr5,
#                                 "scaled_dcg5_in": sdcg5_in,
#                                 "scaled_dcg5": sdcg5,
#                                 "delta_scaled_dcg5": delta_scaled_dcg5,

#                                 # shared-10 metrics
#                                 "dcg10_in": dcg10_in,
#                                 "dcg10": dcg10,
#                                 "delta_dcg10": delta_dcg10,
#                                 "ndcg10_in": ndcg10_in,
#                                 "ndcg10": ndcg10,
#                                 "delta_ndcg10": delta_ndcg10,
#                                 "rr10_in": rr10_in,
#                                 "rr10_out": rr10_out,
#                                 "delta_rr10": delta_rr10,
#                                 "scaled_dcg10_in": sdcg10_in,
#                                 "scaled_dcg10": sdcg10,
#                                 "delta_scaled_dcg10": delta_scaled_dcg10,
#                             })

#             df = pd.DataFrame(records)
#             if df.empty:
#                 print(f"⚠️  No records found for {dataset} × {model}. Skipping.")
#                 continue

#             # significance marks (for all metrics in METRICS_FOR_SIG)
#             sig_marks, sig_lines = compute_sig_marks_and_lines(df, dataset, model)

#             def fmt_basic(metric, mode, size, value, width):
#                 mark = sig_marks.get(metric, {}).get((mode, size), "")
#                 if pd.isna(value):
#                     v = "nan"
#                 else:
#                     v = f"{value:.3f}"
#                 return f"{v}{mark}".rjust(width)

#             def fmt_extra(metric, mode, size, value, width):
#                 mark = sig_marks.get(metric, {}).get((mode, size), "")
#                 if pd.isna(value):
#                     v = "nan"
#                 else:
#                     v = f"{value:.3f}"
#                 return f"{v}{mark}".rjust(width)

#             # ---------- FIRST TABLE (unchanged) ----------
#             summary_basic = (
#                 df.groupby(["mode", "size"])[
#                     ["tau5","rbo5","tau10","rbo10","tau20","rbo20","dcg","ndcg"]
#                 ]
#                 .mean()
#                 .reset_index()
#                 .sort_values(["mode","size"])
#             )

#             with open(OUTPUT_FILE, "w") as out:
#                 out.write("Aggregated τ & RBO over SHARED IDs (5/10/20) + DCG@k + nDCG@k\n")
#                 out.write("=" * 106 + "\n")
#                 out.write(f"Dataset: {dataset}\nModel: {model}\n\n")
#                 out.write(
#                     f"{'Mode':<18}{'Size':<6}"
#                     f"{'τ@5':>8}{'RBO@5':>8}"
#                     f"{'τ@10':>8}{'RBO@10':>8}"
#                     f"{'τ@20':>8}{'RBO@20':>8}"
#                     f"{'DCG@k':>10}{'nDCG@k':>10}\n"
#                 )
#                 out.write("-" * 106 + "\n")
#                 for _, r in summary_basic.iterrows():
#                     mode = r["mode"]
#                     size = int(r["size"])
#                     out.write(
#                         f"{mode:<18} N{size:<5}"
#                         f"{fmt_basic('tau5',  mode, size, r['tau5'],  8)}"
#                         f"{fmt_basic('rbo5',  mode, size, r['rbo5'],  8)}"
#                         f"{fmt_basic('tau10', mode, size, r['tau10'], 8)}"
#                         f"{fmt_basic('rbo10', mode, size, r['rbo10'], 8)}"
#                         f"{fmt_basic('tau20', mode, size, r['tau20'], 8)}"
#                         f"{fmt_basic('rbo20', mode, size, r['rbo20'], 8)}"
#                         f"{fmt_basic('dcg',   mode, size, r['dcg'],  10)}"
#                         f"{fmt_basic('ndcg',  mode, size, r['ndcg'], 10)}\n"
#                     )

#             # ---------- EXTRA TABLE: full-list metrics (existing) ----------
#             summary_extra_full = (
#                 df.groupby(["mode", "size"])[
#                     [
#                         "dcg_in", "dcg", "delta_dcg",
#                         "ndcg_in", "ndcg", "delta_ndcg",
#                         "rr_in", "rr_out", "delta_rr",
#                         "scaled_dcg_in", "scaled_dcg", "delta_scaled_dcg",
#                     ]
#                 ]
#                 .mean()
#                 .reset_index()
#                 .sort_values(["mode", "size"])
#             )

#             with open(OUTPUT_FILE, "a") as out:
#                 out.write("\n\nAdditional metrics vs INPUT ranking (full list, averaged over runs)\n")
#                 out.write("=" * 160 + "\n")
#                 out.write(
#                     f"{'Mode':<18}{'Size':<6}"
#                     f"{'DCG_in':>12}{'DCG_out':>12}{'ΔDCG':>12}"
#                     f"{'nDCG_in':>12}{'nDCG_out':>12}{'ΔnDCG':>12}"
#                     f"{'RR_in':>12}{'RR_out':>12}{'ΔRR':>12}"
#                     f"{'sDCG_in':>12}{'sDCG_out':>12}{'ΔsDCG':>12}\n"
#                 )
#                 out.write("-" * 160 + "\n")
#                 for _, r in summary_extra_full.iterrows():
#                     mode = r["mode"]
#                     size = int(r["size"])
#                     out.write(
#                         f"{mode:<18} N{size:<5}"
#                         f"{fmt_extra('dcg_in',          mode, size, r['dcg_in'],          12)}"
#                         f"{fmt_extra('dcg',             mode, size, r['dcg'],             12)}"
#                         f"{fmt_extra('delta_dcg',       mode, size, r['delta_dcg'],       12)}"
#                         f"{fmt_extra('ndcg_in',         mode, size, r['ndcg_in'],         12)}"
#                         f"{fmt_extra('ndcg',            mode, size, r['ndcg'],            12)}"
#                         f"{fmt_extra('delta_ndcg',      mode, size, r['delta_ndcg'],      12)}"
#                         f"{fmt_extra('rr_in',           mode, size, r['rr_in'],           12)}"
#                         f"{fmt_extra('rr_out',          mode, size, r['rr_out'],          12)}"
#                         f"{fmt_extra('delta_rr',        mode, size, r['delta_rr'],        12)}"
#                         f"{fmt_extra('scaled_dcg_in',   mode, size, r['scaled_dcg_in'],   12)}"
#                         f"{fmt_extra('scaled_dcg',      mode, size, r['scaled_dcg'],      12)}"
#                         f"{fmt_extra('delta_scaled_dcg',mode, size, r['delta_scaled_dcg'],12)}\n"
#                     )

#             # ---------- NEW TABLE: shared-5 metrics ----------
#             summary_shared5 = (
#                 df.groupby(["mode", "size"])[
#                     [
#                         "dcg5_in", "dcg5", "delta_dcg5",
#                         "ndcg5_in", "ndcg5", "delta_ndcg5",
#                         "rr5_in", "rr5_out", "delta_rr5",
#                         "scaled_dcg5_in", "scaled_dcg5", "delta_scaled_dcg5",
#                     ]
#                 ]
#                 .mean()
#                 .reset_index()
#                 .sort_values(["mode", "size"])
#             )

#             with open(OUTPUT_FILE, "a") as out:
#                 out.write("\n\nShared-5 metrics vs INPUT ranking (averaged over runs)\n")
#                 out.write("=" * 160 + "\n")
#                 out.write(
#                     f"{'Mode':<18}{'Size':<6}"
#                     f"{'DCG5_in':>12}{'DCG5_out':>12}{'ΔDCG5':>12}"
#                     f"{'nDCG5_in':>12}{'nDCG5_out':>12}{'ΔnDCG5':>12}"
#                     f"{'RR5_in':>12}{'RR5_out':>12}{'ΔRR5':>12}"
#                     f"{'sDCG5_in':>12}{'sDCG5_out':>12}{'ΔsDCG5':>12}\n"
#                 )
#                 out.write("-" * 160 + "\n")
#                 for _, r in summary_shared5.iterrows():
#                     mode = r["mode"]
#                     size = int(r["size"])
#                     out.write(
#                         f"{mode:<18} N{size:<5}"
#                         f"{fmt_extra('dcg5_in',          mode, size, r['dcg5_in'],          12)}"
#                         f"{fmt_extra('dcg5',             mode, size, r['dcg5'],             12)}"
#                         f"{fmt_extra('delta_dcg5',       mode, size, r['delta_dcg5'],       12)}"
#                         f"{fmt_extra('ndcg5_in',         mode, size, r['ndcg5_in'],         12)}"
#                         f"{fmt_extra('ndcg5',            mode, size, r['ndcg5'],            12)}"
#                         f"{fmt_extra('delta_ndcg5',      mode, size, r['delta_ndcg5'],      12)}"
#                         f"{fmt_extra('rr5_in',           mode, size, r['rr5_in'],           12)}"
#                         f"{fmt_extra('rr5_out',          mode, size, r['rr5_out'],          12)}"
#                         f"{fmt_extra('delta_rr5',        mode, size, r['delta_rr5'],        12)}"
#                         f"{fmt_extra('scaled_dcg5_in',   mode, size, r['scaled_dcg5_in'],   12)}"
#                         f"{fmt_extra('scaled_dcg5',      mode, size, r['scaled_dcg5'],      12)}"
#                         f"{fmt_extra('delta_scaled_dcg5',mode, size, r['delta_scaled_dcg5'],12)}\n"
#                     )

#             # ---------- NEW TABLE: shared-10 metrics ----------
#             summary_shared10 = (
#                 df.groupby(["mode", "size"])[
#                     [
#                         "dcg10_in", "dcg10", "delta_dcg10",
#                         "ndcg10_in", "ndcg10", "delta_ndcg10",
#                         "rr10_in", "rr10_out", "delta_rr10",
#                         "scaled_dcg10_in", "scaled_dcg10", "delta_scaled_dcg10",
#                     ]
#                 ]
#                 .mean()
#                 .reset_index()
#                 .sort_values(["mode", "size"])
#             )

#             with open(OUTPUT_FILE, "a") as out:
#                 out.write("\n\nShared-10 metrics vs INPUT ranking (averaged over runs)\n")
#                 out.write("=" * 160 + "\n")
#                 out.write(
#                     f"{'Mode':<18}{'Size':<6}"
#                     f"{'DCG10_in':>12}{'DCG10_out':>12}{'ΔDCG10':>12}"
#                     f"{'nDCG10_in':>12}{'nDCG10_out':>12}{'ΔnDCG10':>12}"
#                     f"{'RR10_in':>12}{'RR10_out':>12}{'ΔRR10':>12}"
#                     f"{'sDCG10_in':>12}{'sDCG10_out':>12}{'ΔsDCG10':>12}\n"
#                 )
#                 out.write("-" * 160 + "\n")
#                 for _, r in summary_shared10.iterrows():
#                     mode = r["mode"]
#                     size = int(r["size"])
#                     out.write(
#                         f"{mode:<18} N{size:<5}"
#                         f"{fmt_extra('dcg10_in',          mode, size, r['dcg10_in'],          12)}"
#                         f"{fmt_extra('dcg10',             mode, size, r['dcg10'],             12)}"
#                         f"{fmt_extra('delta_dcg10',       mode, size, r['delta_dcg10'],       12)}"
#                         f"{fmt_extra('ndcg10_in',         mode, size, r['ndcg10_in'],         12)}"
#                         f"{fmt_extra('ndcg10',            mode, size, r['ndcg10'],            12)}"
#                         f"{fmt_extra('delta_ndcg10',      mode, size, r['delta_ndcg10'],      12)}"
#                         f"{fmt_extra('rr10_in',           mode, size, r['rr10_in'],           12)}"
#                         f"{fmt_extra('rr10_out',          mode, size, r['rr10_out'],          12)}"
#                         f"{fmt_extra('delta_rr10',        mode, size, r['delta_rr10'],        12)}"
#                         f"{fmt_extra('scaled_dcg10_in',   mode, size, r['scaled_dcg10_in'],   12)}"
#                         f"{fmt_extra('scaled_dcg10',      mode, size, r['scaled_dcg10'],      12)}"
#                         f"{fmt_extra('delta_scaled_dcg10',mode, size, r['delta_scaled_dcg10'],12)}\n"
#                     )

#                 out.write("\nSignificance across sizes (N5, N10, N20) per mode & metric\n")
#                 out.write("Marks: ** best>mid & best>low; * best>mid only or mid>low; † best>low only\n")
#                 for line in sig_lines:
#                     out.write(line + "\n")

#             print(f"✅ Summary (shared-by-IDs) saved to {OUTPUT_FILE}")


            
# compute_all_metrics_shared_by_ids()


############################################################################################################################
####################### V4 - same as above with all metrics and significance but this time for shared Docs #######################
####################### with have ERR and for SDCG we consider the all rel=2  #################
#############################################################################################################################

print("generating summary file\n")

import os, json, numpy as np, pandas as pd
from math import log2
from collections import OrderedDict
from scipy.stats import kendalltau, mannwhitneyu, ttest_ind
from rbo import RankingSimilarity
from tqdm import tqdm
from pyserini.search import get_qrels

def compute_all_metrics_shared_by_ids():
    DATASETS = ["trec-covid", "nfcorpus"]
    MODELS   = ["gpt-3.5", "gpt-4o-mini"]
    MODES    = ["half_relevant", "single_relevant", "single_nonrelevant"]
    SIZES    = [5, 10, 20]
    NUM_INPUTS = 50

    RBO_P = {5: 0.63, 10: 0.79, 20: 0.89}

    # All metrics we test significance across N5/N10/N20
    METRICS_FOR_SIG = [
        # τ/RBO on shared sets
        "tau5", "rbo5", "tau10", "rbo10", "tau20", "rbo20",

        # full-list metrics
        "dcg_in", "dcg", "delta_dcg",
        "idcg_in", "idcg",                # <-- added for IDCG significance
        "ndcg_in", "ndcg", "delta_ndcg",
        "rr_in", "rr_out", "delta_rr",
        "scaled_dcg_in", "scaled_dcg", "delta_scaled_dcg",

        # NEW: ERR (full list)
        "err_in", "err_out", "delta_err",

        # NEW: metrics on shared-5
        "dcg5_in", "dcg5", "delta_dcg5",
        "idcg5_in", "idcg5",             # <-- added for IDCG5 significance
        "ndcg5_in", "ndcg5", "delta_ndcg5",
        "rr5_in", "rr5_out", "delta_rr5",
        "scaled_dcg5_in", "scaled_dcg5", "delta_scaled_dcg5",
        "err5_in", "err5_out", "delta_err5",

        # NEW: metrics on shared-10
        "dcg10_in", "dcg10", "delta_dcg10",
        "idcg10_in", "idcg10",           # <-- added for IDCG10 significance
        "ndcg10_in", "ndcg10", "delta_ndcg10",
        "rr10_in", "rr10_out", "delta_rr10",
        "scaled_dcg10_in", "scaled_dcg10", "delta_scaled_dcg10",
        "err10_in", "err10_out", "delta_err10",
    ]

    # ---------- helpers ----------
    def dcg_at_k(rels, k):
        return sum(rel / log2(i + 2) for i, rel in enumerate(rels[:k]))

    def dcg_for_out(out_docs, qrel_dict, k):
        """Raw DCG@k (no normalization, graded relevance)."""
        if not out_docs:
            return 0.0
        labels = [int(qrel_dict.get(d, 0)) for d in out_docs[:k]]
        return dcg_at_k(labels, k)

    def ideal_dcg_for_out(out_docs, qrel_dict, k):
        """
        Ideal DCG@k for the given set of docs:
        sort the labels for those docs in descending order.
        """
        if not out_docs:
            return 0.0
        labels = [int(qrel_dict.get(d, 0)) for d in out_docs[:k]]
        if not labels:
            return 0.0
        ideal = sorted(labels, reverse=True)
        return dcg_at_k(ideal, k)

    def ndcg_for_out(out_docs, qrel_dict, k):
        """nDCG@k using graded relevance."""
        if not out_docs:
            return 0.0
        labels = [int(qrel_dict.get(d, 0)) for d in out_docs[:k]]
        dcg = dcg_at_k(labels, k)
        ideal = sorted(labels, reverse=True)
        idcg = dcg_at_k(ideal, k)
        return dcg / idcg if idcg > 0 else 0.0

    def reciprocal_rank(out_docs, qrel_dict):
        """RR = 1/rank_of_first_relevant; 0 if no relevant."""
        if not out_docs:
            return 0.0
        for i, d in enumerate(out_docs):
            if int(qrel_dict.get(d, 0)) > 0:
                return 1.0 / (i + 1)
        return 0.0

    def err_for_out(out_docs, qrel_dict, k):
        """
        Expected Reciprocal Rank (ERR)@k with exponential gain.
        Uses graded relevance; gain per rank r:
          R(r) = (2^{rel_r} - 1) / (2^{max_rel} - 1)
        """
        if not out_docs:
            return 0.0
        labels = [int(qrel_dict.get(d, 0)) for d in out_docs[:k]]
        if not labels:
            return 0.0
        max_rel = max(labels)
        if max_rel <= 0:
            return 0.0
        max_gain = (2 ** max_rel - 1)
        if max_gain <= 0:
            return 0.0

        err = 0.0
        p_continue = 1.0
        for rank, rel in enumerate(labels, start=1):
            gain = (2 ** rel - 1) / max_gain
            p_stop = p_continue * gain
            err += p_stop / rank
            p_continue *= (1.0 - gain)
        return err

    def scaled_dcg_for_out(out_docs, qrel_dict, k):
        """
        Scaled DCG@k with graded relevance:
        - Numerator: DCG with graded relevance labels from qrels.
        - Denominator: DCG if all docs in the list (up to k) had rel = 2.
        """
        if not out_docs:
            return 0.0
        labels = [int(qrel_dict.get(d, 0)) for d in out_docs[:k]]
        if not labels:
            return 0.0
        num_dcg = dcg_at_k(labels, k)
        all_twos = [2] * len(labels)
        denom_dcg = dcg_at_k(all_twos, k)
        return num_dcg / denom_dcg if denom_dcg > 0 else 0.0

    def rbo_score_safe(a_ids, b_ids, p):
        a = list(OrderedDict.fromkeys(a_ids or []))
        b = list(OrderedDict.fromkeys(b_ids or []))
        if len(a) < 2 or len(b) < 2:
            return float('nan')
        try:
            return RankingSimilarity(a, b).rbo(p=p)
        except AssertionError:
            return float('nan')

    def rbo_score(a_ids, b_ids, p):
        return rbo_score_safe(a_ids, b_ids, p)

    def load_input_ids(root, size, mode, qid, inp_idx):
        f = os.path.join(root, f"N{size}", mode, str(qid), f"in{inp_idx:02d}_run0.json")
        if not os.path.exists(f):
            return None
        try:
            return json.load(open(f))["input_docids"]
        except Exception:
            return None

    def load_output_ids(root, size, mode, qid, inp_idx, run):
        f = os.path.join(root, f"N{size}", mode, str(qid), f"in{inp_idx:02d}_run{run}.json")
        if not os.path.exists(f):
            return None
        try:
            return json.load(open(f))["output_docids"]
        except Exception:
            return None

    def kendall_tau_for_shared(a_list, b_list, shared_ids):
        if not shared_ids:
            return np.nan
        a_filtered = [d for d in a_list if d in shared_ids]
        b_filtered = [d for d in b_list if d in shared_ids]
        n = len(a_filtered)
        if n < 2 or len(b_filtered) != n:
            return np.nan
        a_pos = {d: i for i, d in enumerate(a_filtered)}
        seq_a = [a_pos[d] for d in b_filtered if d in a_pos]
        seq_b = list(range(len(seq_a)))
        if len(seq_a) < 2:
            return np.nan
        tau, _ = kendalltau(seq_a, seq_b)
        return tau

    def significant(a, b):
        """
        Significance decision between two samples a, b using:
        - Mann–Whitney U
        - Welch's t-test
        Returns True if both tests have p < 0.05.
        """
        a = np.asarray(a)
        b = np.asarray(b)
        if len(a) < 2 or len(b) < 2:
            return False, np.nan, np.nan
        p_u = mannwhitneyu(a, b, alternative="two-sided").pvalue
        p_t = ttest_ind(a, b, equal_var=False).pvalue
        return (p_u < 0.05 and p_t < 0.05), p_u, p_t

    def compute_sig_marks_and_lines(df, dataset, model):
        """
        For each mode & metric, compare the three sizes N5/N10/N20.
        Returns:
          marks[metric][(mode, size)] -> mark string ("", "*", "†", "**")
          lines -> list of text lines summarising significance decisions.
        """
        marks = {m: {} for m in METRICS_FOR_SIG}
        lines = []

        for mode in sorted(df["mode"].unique()):
            df_mode = df[df["mode"] == mode]
            for metric in METRICS_FOR_SIG:
                if metric not in df_mode.columns:
                    continue
                vals = {}
                means = {}
                for s in SIZES:
                    arr = df_mode[df_mode["size"] == s][metric].dropna().to_numpy()
                    if arr.size > 0:
                        vals[s] = arr
                        means[s] = float(np.mean(arr))

                if len(vals) < 3:
                    continue

                ordered = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
                best_size, best_mean = ordered[0]
                mid_size, mid_mean   = ordered[1]
                low_size, low_mean   = ordered[2]

                sig_bm, p_u_bm, p_t_bm = significant(vals[best_size], vals[mid_size])
                sig_bl, p_u_bl, p_t_bl = significant(vals[best_size], vals[low_size])
                sig_ml, p_u_ml, p_t_ml = significant(vals[mid_size],  vals[low_size])

                mark_best = ""
                if sig_bm and sig_bl:
                    mark_best = "**"
                elif sig_bm and not sig_bl:
                    mark_best = "*"
                elif (not sig_bm) and sig_bl:
                    mark_best = "†"

                if mark_best:
                    marks[metric][(mode, best_size)] = mark_best

                if sig_ml:
                    prev = marks[metric].get((mode, mid_size), "")
                    marks[metric][(mode, mid_size)] = prev + "*"

                line = (
                    f"{dataset} | {model} | mode={mode} | metric={metric} | "
                    f"best=N{best_size} (mean={best_mean:.6f}), "
                    f"mid=N{mid_size} (mean={mid_mean:.6f}), "
                    f"low=N{low_size} (mean={low_mean:.6f}) | "
                    f"sig(best>mid)={sig_bm}, sig(best>low)={sig_bl}, sig(mid>low)={sig_ml}"
                )
                lines.append(line)

        return marks, lines

    def compute_shared_metrics(doc_list, shared_ids, qrel_dict, k_target):
        """
        Restrict doc_list to shared_ids (preserve order) and compute
        DCG@k_target, IDCG@k_target, nDCG@k_target, RR, ERR, scaled DCG.
        Returns (dcg, idcg, ndcg, rr, err, sdcg). NaNs if no shared docs.
        """
        if not doc_list or not shared_ids:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        restricted = [d for d in doc_list if d in shared_ids]
        if len(restricted) == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        dcg = dcg_for_out(restricted, qrel_dict, k_target)
        idcg = ideal_dcg_for_out(restricted, qrel_dict, k_target)
        ndcg = ndcg_for_out(restricted, qrel_dict, k_target)
        rr = reciprocal_rank(restricted, qrel_dict)
        err = err_for_out(restricted, qrel_dict, k_target)
        sdcg = scaled_dcg_for_out(restricted, qrel_dict, k_target)
        return dcg, idcg, ndcg, rr, err, sdcg

    # ---------- main loop ----------
    for dataset in DATASETS:
        TOPICS_NAME = "beir-v1.0.0-trec-covid-test" if dataset == "trec-covid" else "beir-v1.0.0-nfcorpus-test"
        qrels = get_qrels(TOPICS_NAME)

        for model in MODELS:
            RESULTS_ROOT = f"results/{dataset}_{model}"
            if not os.path.isdir(RESULTS_ROOT):
                print(f"⚠️  Skipping {RESULTS_ROOT} (folder not found)")
                continue

            OUTPUT_FILE = os.path.join(RESULTS_ROOT, "results_summary_shared_ids.txt")
            print(f"\n=== Processing {dataset} × {model} (shared-by-IDs) ===")

            records = []
            size_dirs = sorted(
                [d for d in os.listdir(RESULTS_ROOT) if d.startswith("N")],
                key=lambda x: int(x[1:])
            )

            for size_dir in size_dirs:
                k = int(size_dir[1:])
                for mode in MODES:
                    mode_dir = os.path.join(RESULTS_ROOT, size_dir, mode)
                    if not os.path.isdir(mode_dir):
                        continue
                    try:
                        qids = sorted(os.listdir(mode_dir))
                    except Exception:
                        continue

                    for qid in tqdm(qids, desc=f"{dataset}_{model} {mode} N{k}"):
                        for inp_idx in range(NUM_INPUTS):
                            in5  = load_input_ids(RESULTS_ROOT, 5,  mode, qid, inp_idx) if os.path.isdir(os.path.join(RESULTS_ROOT,"N5", mode, str(qid)))  else None
                            in10 = load_input_ids(RESULTS_ROOT, 10, mode, qid, inp_idx) if os.path.isdir(os.path.join(RESULTS_ROOT,"N10",mode, str(qid))) else None
                            in20 = load_input_ids(RESULTS_ROOT, 20, mode, qid, inp_idx) if os.path.isdir(os.path.join(RESULTS_ROOT,"N20",mode, str(qid))) else None

                            shared5  = set(in5).intersection(in10, in20) if (in5 and in10 and in20) else set()
                            shared10 = set(in10).intersection(in20)       if (in10 and in20)       else set()
                            shared20 = set(in20) if in20 else set()

                            out0 = load_output_ids(RESULTS_ROOT, k, mode, qid, inp_idx, 0)
                            out1 = load_output_ids(RESULTS_ROOT, k, mode, qid, inp_idx, 1)
                            if not out0 or not out1:
                                continue

                            # τ / RBO on shared sets
                            tau5  = kendall_tau_for_shared(out0, out1, shared5)  if shared5  else np.nan
                            rbo5  = rbo_score([d for d in out0 if d in shared5],
                                              [d for d in out1 if d in shared5], RBO_P[5])  if shared5  else np.nan

                            tau10 = kendall_tau_for_shared(out0, out1, shared10) if shared10 else np.nan
                            rbo10 = rbo_score([d for d in out0 if d in shared10],
                                              [d for d in out1 if d in shared10], RBO_P[10]) if shared10 else np.nan

                            tau20 = kendall_tau_for_shared(out0, out1, shared20) if shared20 else np.nan
                            rbo20 = rbo_score([d for d in out0 if d in shared20],
                                              [d for d in out1 if d in shared20], RBO_P[20]) if shared20 else np.nan

                            # qrels
                            qrel_dict = get_qrel_dict_robust(qrels, qid)

                            # full-list output metrics @k (existing + IDCG + ERR)
                            dcg0   = dcg_for_out(out0, qrel_dict, k)
                            dcg1   = dcg_for_out(out1, qrel_dict, k)
                            dcg_mean = float(np.mean([dcg0, dcg1]))

                            idcg0  = ideal_dcg_for_out(out0, qrel_dict, k)
                            idcg1  = ideal_dcg_for_out(out1, qrel_dict, k)
                            idcg_mean = float(np.mean([idcg0, idcg1]))

                            ndcg0  = ndcg_for_out(out0, qrel_dict, k)
                            ndcg1  = ndcg_for_out(out1, qrel_dict, k)
                            ndcg_mean = float(np.mean([ndcg0, ndcg1]))

                            rr0 = reciprocal_rank(out0, qrel_dict)
                            rr1 = reciprocal_rank(out1, qrel_dict)
                            rr_out_mean = float(np.mean([rr0, rr1]))

                            err0 = err_for_out(out0, qrel_dict, k)
                            err1 = err_for_out(out1, qrel_dict, k)
                            err_out_mean = float(np.mean([err0, err1]))

                            scaled_dcg0 = scaled_dcg_for_out(out0, qrel_dict, k)
                            scaled_dcg1 = scaled_dcg_for_out(out1, qrel_dict, k)
                            scaled_dcg_mean = float(np.mean([scaled_dcg0, scaled_dcg1]))

                            # choose input docs for this size
                            if k == 5:
                                inp_docs = in5
                            elif k == 10:
                                inp_docs = in10
                            elif k == 20:
                                inp_docs = in20
                            else:
                                inp_docs = None

                            # base metrics vs input (full list)
                            if inp_docs:
                                dcg_in = dcg_for_out(inp_docs, qrel_dict, k)
                                idcg_in = ideal_dcg_for_out(inp_docs, qrel_dict, k)
                                ndcg_in = ndcg_for_out(inp_docs, qrel_dict, k)
                                rr_in = reciprocal_rank(inp_docs, qrel_dict)
                                err_in = err_for_out(inp_docs, qrel_dict, k)
                                scaled_dcg_in = scaled_dcg_for_out(inp_docs, qrel_dict, k)

                                delta_dcg = dcg_mean - dcg_in
                                delta_ndcg = ndcg_mean - ndcg_in
                                delta_rr = rr_out_mean - rr_in
                                delta_err = err_out_mean - err_in
                                delta_scaled_dcg = scaled_dcg_mean - scaled_dcg_in
                            else:
                                dcg_in = idcg_in = ndcg_in = rr_in = err_in = scaled_dcg_in = np.nan
                                delta_dcg = delta_ndcg = delta_rr = delta_err = delta_scaled_dcg = np.nan

                            # ===== NEW: metrics on shared-5 & shared-10 =====
                            # shared-5 metrics (if shared5 non-empty and we have input)
                            if inp_docs and shared5:
                                dcg5_in, idcg5_in, ndcg5_in, rr5_in, err5_in, sdcg5_in = compute_shared_metrics(
                                    inp_docs, shared5, qrel_dict, 5
                                )
                                dcg5_0, idcg5_0, ndcg5_0, rr5_0, err5_0, sdcg5_0 = compute_shared_metrics(
                                    out0, shared5, qrel_dict, 5
                                )
                                dcg5_1, idcg5_1, ndcg5_1, rr5_1, err5_1, sdcg5_1 = compute_shared_metrics(
                                    out1, shared5, qrel_dict, 5
                                )
                                dcg5 = float(np.nanmean([dcg5_0, dcg5_1]))
                                idcg5 = float(np.nanmean([idcg5_0, idcg5_1]))
                                ndcg5 = float(np.nanmean([ndcg5_0, ndcg5_1]))
                                rr5_out = float(np.nanmean([rr5_0, rr5_1]))
                                err5_out = float(np.nanmean([err5_0, err5_1]))
                                sdcg5 = float(np.nanmean([sdcg5_0, sdcg5_1]))

                                delta_dcg5 = dcg5 - dcg5_in
                                delta_ndcg5 = ndcg5 - ndcg5_in
                                delta_rr5 = rr5_out - rr5_in
                                delta_err5 = err5_out - err5_in
                                delta_scaled_dcg5 = sdcg5 - sdcg5_in
                            else:
                                dcg5_in = idcg5_in = ndcg5_in = rr5_in = err5_in = sdcg5_in = np.nan
                                dcg5 = idcg5 = ndcg5 = rr5_out = err5_out = sdcg5 = np.nan
                                delta_dcg5 = delta_ndcg5 = delta_rr5 = delta_err5 = delta_scaled_dcg5 = np.nan

                            # shared-10 metrics (if shared10 non-empty and we have input)
                            if inp_docs and shared10:
                                dcg10_in, idcg10_in, ndcg10_in, rr10_in, err10_in, sdcg10_in = compute_shared_metrics(
                                    inp_docs, shared10, qrel_dict, 10
                                )
                                dcg10_0, idcg10_0, ndcg10_0, rr10_0, err10_0, sdcg10_0 = compute_shared_metrics(
                                    out0, shared10, qrel_dict, 10
                                )
                                dcg10_1, idcg10_1, ndcg10_1, rr10_1, err10_1, sdcg10_1 = compute_shared_metrics(
                                    out1, shared10, qrel_dict, 10
                                )
                                dcg10 = float(np.nanmean([dcg10_0, dcg10_1]))
                                idcg10 = float(np.nanmean([idcg10_0, idcg10_1]))
                                ndcg10 = float(np.nanmean([ndcg10_0, ndcg10_1]))
                                rr10_out = float(np.nanmean([rr10_0, rr10_1]))
                                err10_out = float(np.nanmean([err10_0, err10_1]))
                                sdcg10 = float(np.nanmean([sdcg10_0, sdcg10_1]))

                                delta_dcg10 = dcg10 - dcg10_in
                                delta_ndcg10 = ndcg10 - ndcg10_in
                                delta_rr10 = rr10_out - rr10_in
                                delta_err10 = err10_out - err10_in
                                delta_scaled_dcg10 = sdcg10 - sdcg10_in
                            else:
                                dcg10_in = idcg10_in = ndcg10_in = rr10_in = err10_in = sdcg10_in = np.nan
                                dcg10 = idcg10 = ndcg10 = rr10_out = err10_out = sdcg10 = np.nan
                                delta_dcg10 = delta_ndcg10 = delta_rr10 = delta_err10 = delta_scaled_dcg10 = np.nan

                            records.append({
                                "mode": mode,
                                "size": k,

                                "tau5": tau5, "rbo5": rbo5,
                                "tau10": tau10, "rbo10": rbo10,
                                "tau20": tau20, "rbo20": rbo20,

                                # full-list metrics
                                "dcg": dcg_mean,
                                "idcg": idcg_mean,
                                "ndcg": ndcg_mean,
                                "dcg_in": dcg_in,
                                "idcg_in": idcg_in,
                                "ndcg_in": ndcg_in,
                                "delta_dcg": delta_dcg,
                                "delta_ndcg": delta_ndcg,
                                "rr_in": rr_in,
                                "rr_out": rr_out_mean,
                                "delta_rr": delta_rr,
                                "err_in": err_in,
                                "err_out": err_out_mean,
                                "delta_err": delta_err,
                                "scaled_dcg_in": scaled_dcg_in,
                                "scaled_dcg": scaled_dcg_mean,
                                "delta_scaled_dcg": delta_scaled_dcg,

                                # shared-5 metrics
                                "dcg5_in": dcg5_in,
                                "dcg5": dcg5,
                                "delta_dcg5": delta_dcg5,
                                "idcg5_in": idcg5_in,
                                "idcg5": idcg5,
                                "ndcg5_in": ndcg5_in,
                                "ndcg5": ndcg5,
                                "delta_ndcg5": delta_ndcg5,
                                "rr5_in": rr5_in,
                                "rr5_out": rr5_out,
                                "delta_rr5": delta_rr5,
                                "err5_in": err5_in,
                                "err5_out": err5_out,
                                "delta_err5": delta_err5,
                                "scaled_dcg5_in": sdcg5_in,
                                "scaled_dcg5": sdcg5,
                                "delta_scaled_dcg5": delta_scaled_dcg5,

                                # shared-10 metrics
                                "dcg10_in": dcg10_in,
                                "dcg10": dcg10,
                                "delta_dcg10": delta_dcg10,
                                "idcg10_in": idcg10_in,
                                "idcg10": idcg10,
                                "ndcg10_in": ndcg10_in,
                                "ndcg10": ndcg10,
                                "delta_ndcg10": delta_ndcg10,
                                "rr10_in": rr10_in,
                                "rr10_out": rr10_out,
                                "delta_rr10": delta_rr10,
                                "err10_in": err10_in,
                                "err10_out": err10_out,
                                "delta_err10": delta_err10,
                                "scaled_dcg10_in": sdcg10_in,
                                "scaled_dcg10": sdcg10,
                                "delta_scaled_dcg10": delta_scaled_dcg10,
                            })

            df = pd.DataFrame(records)
            if df.empty:
                print(f"⚠️  No records found for {dataset} × {model}. Skipping.")
                continue

            # significance marks (for all metrics in METRICS_FOR_SIG)
            sig_marks, sig_lines = compute_sig_marks_and_lines(df, dataset, model)

            def fmt_basic(metric, mode, size, value, width):
                mark = sig_marks.get(metric, {}).get((mode, size), "")
                if pd.isna(value):
                    v = "nan"
                else:
                    v = f"{value:.6f}"
                return f"{v}{mark}".rjust(width)

            def fmt_extra(metric, mode, size, value, width):
                mark = sig_marks.get(metric, {}).get((mode, size), "")
                if pd.isna(value):
                    v = "nan"
                else:
                    v = f"{value:.6f}"
                return f"{v}{mark}".rjust(width)

            # ---------- FIRST TABLE (unchanged) ----------
            summary_basic = (
                df.groupby(["mode", "size"])[
                    ["tau5","rbo5","tau10","rbo10","tau20","rbo20","dcg","ndcg"]
                ]
                .mean()
                .reset_index()
                .sort_values(["mode","size"])
            )

            with open(OUTPUT_FILE, "w") as out:
                out.write("Aggregated τ & RBO over SHARED IDs (5/10/20) + DCG@k + nDCG@k\n")
                out.write("=" * 106 + "\n")
                out.write(f"Dataset: {dataset}\nModel: {model}\n\n")
                out.write(
                    f"{'Mode':<18}{'Size':<6}"
                    f"{'τ@5':>8}{'RBO@5':>8}"
                    f"{'τ@10':>8}{'RBO@10':>8}"
                    f"{'τ@20':>8}{'RBO@20':>8}"
                    f"{'DCG@k':>10}{'nDCG@k':>10}\n"
                )
                out.write("-" * 106 + "\n")
                for _, r in summary_basic.iterrows():
                    mode = r["mode"]
                    size = int(r["size"])
                    out.write(
                        f"{mode:<18} N{size:<5}"
                        f"{fmt_basic('tau5',  mode, size, r['tau5'],  8)}"
                        f"{fmt_basic('rbo5',  mode, size, r['rbo5'],  8)}"
                        f"{fmt_basic('tau10', mode, size, r['tau10'], 8)}"
                        f"{fmt_basic('rbo10', mode, size, r['rbo10'], 8)}"
                        f"{fmt_basic('tau20', mode, size, r['tau20'], 8)}"
                        f"{fmt_basic('rbo20', mode, size, r['rbo20'], 8)}"
                        f"{fmt_basic('dcg',   mode, size, r['dcg'],  10)}"
                        f"{fmt_basic('ndcg',  mode, size, r['ndcg'], 10)}\n"
                    )

            # ---------- EXTRA TABLE: full-list metrics ----------
            summary_extra_full = (
                df.groupby(["mode", "size"])[
                    [
                        "dcg_in", "dcg", "delta_dcg",
                        "idcg_in", "idcg",
                        "ndcg_in", "ndcg", "delta_ndcg",
                        "rr_in", "rr_out", "delta_rr",
                        "err_in", "err_out", "delta_err",
                        "scaled_dcg_in", "scaled_dcg", "delta_scaled_dcg",
                    ]
                ]
                .mean()
                .reset_index()
                .sort_values(["mode", "size"])
            )

            with open(OUTPUT_FILE, "a") as out:
                out.write("\n\nAdditional metrics vs INPUT ranking (full list, averaged over runs)\n")
                out.write("=" * 220 + "\n")
                out.write(
                    f"{'Mode':<18}{'Size':<6}"
                    f"{'DCG_in':>12}{'DCG_out':>12}{'ΔDCG':>12}"
                    f"{'IDCG_in':>12}{'IDCG_out':>12}"
                    f"{'nDCG_in':>12}{'nDCG_out':>12}{'ΔnDCG':>12}"
                    f"{'RR_in':>12}{'RR_out':>12}{'ΔRR':>12}"
                    f"{'ERR_in':>12}{'ERR_out':>12}{'ΔERR':>12}"
                    f"{'sDCG_in':>12}{'sDCG_out':>12}{'ΔsDCG':>12}\n"
                )
                out.write("-" * 220 + "\n")
                for _, r in summary_extra_full.iterrows():
                    mode = r["mode"]
                    size = int(r["size"])
                    out.write(
                        f"{mode:<18} N{size:<5}"
                        f"{fmt_extra('dcg_in',          mode, size, r['dcg_in'],          12)}"
                        f"{fmt_extra('dcg',             mode, size, r['dcg'],             12)}"
                        f"{fmt_extra('delta_dcg',       mode, size, r['delta_dcg'],       12)}"
                        f"{fmt_extra('idcg_in',         mode, size, r['idcg_in'],         12)}"
                        f"{fmt_extra('idcg',            mode, size, r['idcg'],            12)}"
                        f"{fmt_extra('ndcg_in',         mode, size, r['ndcg_in'],         12)}"
                        f"{fmt_extra('ndcg',            mode, size, r['ndcg'],            12)}"
                        f"{fmt_extra('delta_ndcg',      mode, size, r['delta_ndcg'],      12)}"
                        f"{fmt_extra('rr_in',           mode, size, r['rr_in'],           12)}"
                        f"{fmt_extra('rr_out',          mode, size, r['rr_out'],          12)}"
                        f"{fmt_extra('delta_rr',        mode, size, r['delta_rr'],        12)}"
                        f"{fmt_extra('err_in',          mode, size, r['err_in'],          12)}"
                        f"{fmt_extra('err_out',         mode, size, r['err_out'],         12)}"
                        f"{fmt_extra('delta_err',       mode, size, r['delta_err'],       12)}"
                        f"{fmt_extra('scaled_dcg_in',   mode, size, r['scaled_dcg_in'],   12)}"
                        f"{fmt_extra('scaled_dcg',      mode, size, r['scaled_dcg'],      12)}"
                        f"{fmt_extra('delta_scaled_dcg',mode, size, r['delta_scaled_dcg'],12)}\n"
                    )

            # ---------- NEW TABLE: shared-5 metrics ----------
            summary_shared5 = (
                df.groupby(["mode", "size"])[
                    [
                        "dcg5_in", "dcg5", "delta_dcg5",
                        "idcg5_in", "idcg5",
                        "ndcg5_in", "ndcg5", "delta_ndcg5",
                        "rr5_in", "rr5_out", "delta_rr5",
                        "err5_in", "err5_out", "delta_err5",
                        "scaled_dcg5_in", "scaled_dcg5", "delta_scaled_dcg5",
                    ]
                ]
                .mean()
                .reset_index()
                .sort_values(["mode", "size"])
            )

            with open(OUTPUT_FILE, "a") as out:
                out.write("\n\nShared-5 metrics vs INPUT ranking (averaged over runs)\n")
                out.write("=" * 220 + "\n")
                out.write(
                    f"{'Mode':<18}{'Size':<6}"
                    f"{'DCG5_in':>12}{'DCG5_out':>12}{'ΔDCG5':>12}"
                    f"{'IDCG5_in':>12}{'IDCG5_out':>12}"
                    f"{'nDCG5_in':>12}{'nDCG5_out':>12}{'ΔnDCG5':>12}"
                    f"{'RR5_in':>12}{'RR5_out':>12}{'ΔRR5':>12}"
                    f"{'ERR5_in':>12}{'ERR5_out':>12}{'ΔERR5':>12}"
                    f"{'sDCG5_in':>12}{'sDCG5_out':>12}{'ΔsDCG5':>12}\n"
                )
                out.write("-" * 220 + "\n")
                for _, r in summary_shared5.iterrows():
                    mode = r["mode"]
                    size = int(r["size"])
                    out.write(
                        f"{mode:<18} N{size:<5}"
                        f"{fmt_extra('dcg5_in',          mode, size, r['dcg5_in'],          12)}"
                        f"{fmt_extra('dcg5',             mode, size, r['dcg5'],             12)}"
                        f"{fmt_extra('delta_dcg5',       mode, size, r['delta_dcg5'],       12)}"
                        f"{fmt_extra('idcg5_in',         mode, size, r['idcg5_in'],         12)}"
                        f"{fmt_extra('idcg5',            mode, size, r['idcg5'],            12)}"
                        f"{fmt_extra('ndcg5_in',         mode, size, r['ndcg5_in'],         12)}"
                        f"{fmt_extra('ndcg5',            mode, size, r['ndcg5'],            12)}"
                        f"{fmt_extra('delta_ndcg5',      mode, size, r['delta_ndcg5'],      12)}"
                        f"{fmt_extra('rr5_in',           mode, size, r['rr5_in'],           12)}"
                        f"{fmt_extra('rr5_out',          mode, size, r['rr5_out'],          12)}"
                        f"{fmt_extra('delta_rr5',        mode, size, r['delta_rr5'],        12)}"
                        f"{fmt_extra('err5_in',          mode, size, r['err5_in'],          12)}"
                        f"{fmt_extra('err5_out',         mode, size, r['err5_out'],         12)}"
                        f"{fmt_extra('delta_err5',       mode, size, r['delta_err5'],       12)}"
                        f"{fmt_extra('scaled_dcg5_in',   mode, size, r['scaled_dcg5_in'],   12)}"
                        f"{fmt_extra('scaled_dcg5',      mode, size, r['scaled_dcg5'],      12)}"
                        f"{fmt_extra('delta_scaled_dcg5',mode, size, r['delta_scaled_dcg5'],12)}\n"
                    )

            # ---------- NEW TABLE: shared-10 metrics ----------
            summary_shared10 = (
                df.groupby(["mode", "size"])[
                    [
                        "dcg10_in", "dcg10", "delta_dcg10",
                        "idcg10_in", "idcg10",
                        "ndcg10_in", "ndcg10", "delta_ndcg10",
                        "rr10_in", "rr10_out", "delta_rr10",
                        "err10_in", "err10_out", "delta_err10",
                        "scaled_dcg10_in", "scaled_dcg10", "delta_scaled_dcg10",
                    ]
                ]
                .mean()
                .reset_index()
                .sort_values(["mode", "size"])
            )

            with open(OUTPUT_FILE, "a") as out:
                out.write("\n\nShared-10 metrics vs INPUT ranking (averaged over runs)\n")
                out.write("=" * 220 + "\n")
                out.write(
                    f"{'Mode':<18}{'Size':<6}"
                    f"{'DCG10_in':>12}{'DCG10_out':>12}{'ΔDCG10':>12}"
                    f"{'IDCG10_in':>12}{'IDCG10_out':>12}"
                    f"{'nDCG10_in':>12}{'nDCG10_out':>12}{'ΔnDCG10':>12}"
                    f"{'RR10_in':>12}{'RR10_out':>12}{'ΔRR10':>12}"
                    f"{'ERR10_in':>12}{'ERR10_out':>12}{'ΔERR10':>12}"
                    f"{'sDCG10_in':>12}{'sDCG10_out':>12}{'ΔsDCG10':>12}\n"
                )
                out.write("-" * 220 + "\n")
                for _, r in summary_shared10.iterrows():
                    mode = r["mode"]
                    size = int(r["size"])
                    out.write(
                        f"{mode:<18} N{size:<5}"
                        f"{fmt_extra('dcg10_in',          mode, size, r['dcg10_in'],          12)}"
                        f"{fmt_extra('dcg10',             mode, size, r['dcg10'],             12)}"
                        f"{fmt_extra('delta_dcg10',       mode, size, r['delta_dcg10'],       12)}"
                        f"{fmt_extra('idcg10_in',         mode, size, r['idcg10_in'],         12)}"
                        f"{fmt_extra('idcg10',            mode, size, r['idcg10'],            12)}"
                        f"{fmt_extra('ndcg10_in',         mode, size, r['ndcg10_in'],         12)}"
                        f"{fmt_extra('ndcg10',            mode, size, r['ndcg10'],            12)}"
                        f"{fmt_extra('delta_ndcg10',      mode, size, r['delta_ndcg10'],      12)}"
                        f"{fmt_extra('rr10_in',           mode, size, r['rr10_in'],           12)}"
                        f"{fmt_extra('rr10_out',          mode, size, r['rr10_out'],          12)}"
                        f"{fmt_extra('delta_rr10',        mode, size, r['delta_rr10'],        12)}"
                        f"{fmt_extra('err10_in',          mode, size, r['err10_in'],          12)}"
                        f"{fmt_extra('err10_out',         mode, size, r['err10_out'],         12)}"
                        f"{fmt_extra('delta_err10',       mode, size, r['delta_err10'],       12)}"
                        f"{fmt_extra('scaled_dcg10_in',   mode, size, r['scaled_dcg10_in'],   12)}"
                        f"{fmt_extra('scaled_dcg10',      mode, size, r['scaled_dcg10'],      12)}"
                        f"{fmt_extra('delta_scaled_dcg10',mode, size, r['delta_scaled_dcg10'],12)}\n"
                    )

                out.write("\nSignificance across sizes (N5, N10, N20) per mode & metric\n")
                out.write("Marks: ** best>mid & best>low; * best>mid only or mid>low; † best>low only\n")
                for line in sig_lines:
                    out.write(line + "\n")

            print(f"✅ Summary (shared-by-IDs) saved to {OUTPUT_FILE}")


# #### this is a current and true version of this function for sigir 2026 paper ########
# compute_all_metrics_shared_by_ids()





##############################################################################################################################
############################################# check number of relevant doc in each mode ######################################
##############################################################################################################################

import os, json
from pyserini.search import get_qrels


def get_qrel_dict_robust(qrels_raw, qid):
    """
    Return the per-qid qrels dict, handling int/str key mismatches.
    Tries: exact, int(qid), str(qid); returns {} if none found.
    """
    if qid in qrels_raw:
        return qrels_raw[qid]
    try:
        qi = int(qid)
        if qi in qrels_raw:
            return qrels_raw[qi]
    except Exception:
        pass
    qs = str(qid)
    if qs in qrels_raw:
        return qrels_raw[qs]
    return {}


def _count_relevant(docids, qrel_dict):
    """Return number of relevant docs (label > 0) among docids."""
    return sum(1 for d in docids if int(qrel_dict.get(d, 0)) > 0)

def audit_input_composition(
    results_root: str,
    topics_name: str,
    modes = ("single_nonrelevant","half_relevant","single_relevant"),
    sizes = (5,10,20),
    num_inputs: int = 50,
    runs = (0,1),
    dataset_short: str = "trec-covid",
    model_short: str = "gpt-4o-mini",
):
    """
    Walks results_root/N{size}/{mode}/{qid}/ and prints, for each input index:
      dataset  model  mode  N{size}  QID={qid}  Input={i}  AvgRelDocs=X  Runs=[r0,r1,...]
    Uses input_docids from each run log.
    """
    qrels = get_qrels(topics_name)  # dict: {qid -> {docid: label, ...}}

    # quick sanity on available sizes/modes
    for size in sizes:
        for mode in modes:
            mode_dir = os.path.join(results_root, f"N{size}", mode)
            if not os.path.isdir(mode_dir):
                # silent skip to keep output clean, like your previous runs
                continue

            # qids are subfolders; don't assume numeric
            qids = sorted(os.listdir(mode_dir))
            for qid in qids:
                qdir = os.path.join(mode_dir, qid)
                if not os.path.isdir(qdir):
                    continue
                qrel_dict = get_qrel_dict_robust(qrels, qid)

                # (Optional) if qrel_dict is empty, you may still want to print that info:
                # if not qrel_dict: print(f"⚠️ QID={qid} has no qrels; skipping."); continue

                for inp_idx in range(num_inputs):
                    per_run_counts = []
                    had_any = False
                    for r in runs:
                        fpath = os.path.join(qdir, f"in{inp_idx:02d}_run{r}.json")
                        if not os.path.exists(fpath):
                            per_run_counts.append(None)
                            continue
                        try:
                            log = json.load(open(fpath))
                            in_ids = log.get("input_docids", [])
                            rel_cnt = _count_relevant(in_ids, qrel_dict)
                            per_run_counts.append(rel_cnt)
                            had_any = True
                        except Exception:
                            per_run_counts.append(None)

                    if not had_any:
                        # no files for this input across runs; keep quiet
                        continue

                    # compute avg over available runs (ignore None)
                    vals = [v for v in per_run_counts if v is not None]
                    avg_rel = sum(vals)/len(vals) if vals else 0.0
                    # pretty print run list with None as  "-"
                    runs_str = "[" + ", ".join("-" if v is None else str(v) for v in per_run_counts) + "]"

                    print(
                        f"{dataset_short}\t{model_short}\t{mode}\tN{size}\t"
                        f"QID={qid}\tInput={inp_idx}\tAvgRelDocs={avg_rel:.1f}\tRuns={runs_str}"
                    )

# ── Example calls (pick the right one) ─────────────────────────
# TREC-COVID:
# audit_input_composition(
#     results_root="results/trec-covid_gpt-4o-mini",
#     topics_name="beir-v1.0.0-trec-covid-test",
#     modes=("half_relevant",),
#     #,"single_nonrelevant","half_relevant","single_relevant"
#     sizes=(5,10,20),
#     num_inputs=50,
#     runs=(0,1),
#     dataset_short="trec-covid",
#     model_short="gpt-4o-mini",
# )


# audit_input_composition(
#     results_root="results/trec-covid_gpt-3.5",
#     topics_name="beir-v1.0.0-trec-covid-test",
#     modes=("single_nonrelevant","half_relevant","single_relevant"),
#     #,"single_nonrelevant","half_relevant","single_relevant"
#     sizes=(5,10,20),
#     num_inputs=50,
#     runs=(0,1),
#     dataset_short="trec-covid",
#     model_short="gpt-3.5",
# )

# NF-Corpus:
# audit_input_composition(
#     results_root="results/nfcorpus_gpt-4o-mini",
#     topics_name="beir-v1.0.0-nfcorpus-test",
#     modes=("single_nonrelevant","half_relevant","single_relevant"),
#     sizes=(5,10,20),
#     num_inputs=50,
#     runs=(0,1),
#     dataset_short="nfcorpus",
#     model_short="gpt-4o-mini",
# )



# NF-Corpus:
# audit_input_composition(
#     results_root="results/nfcorpus_gpt-3.5",
#     topics_name="beir-v1.0.0-nfcorpus-test",
#     modes=("single_nonrelevant","half_relevant","single_relevant"),
#     sizes=(5,10,20),
#     num_inputs=50,
#     runs=(0,1),
#     dataset_short="nfcorpus",
#     model_short="gpt-3.5",
# )



#############################################################################################################################
############################ meet the condition that 5 subset 10 subset 20 #################################################
#############################################################################################################################

import os, json
from collections import defaultdict

def _safe_load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

def print_subset_chain_ok(
    dataset_short="trec-covid",           # "trec-covid" or "nfcorpus"
    model_short="gpt-4o-mini",            # "gpt-3.5" or "gpt-4o-mini" (short form used in your folders)
    modes=("single_relevant","single_nonrelevant","half_relevant"),
    num_inputs=50
):
    """
    Print ONLY (mode, qid, input_index) triplets where inputs satisfy:
        N5 ⊆ N10 ⊆ N20
    """
    root = f"results/{dataset_short}_{model_short}"
    if not os.path.isdir(root):
        print(f"!! Missing results root: {root}")
        return

    def in_path(k, mode, qid, i):
        return os.path.join(root, f"N{k}", mode, str(qid), f"in{i:02d}_run0.json")

    printed = 0
    per_mode_counts = defaultdict(int)

    for mode in modes:
        # collect all qids that exist under any N-size for this mode
        qids = set()
        for k in (5,10,20):
            mode_dir = os.path.join(root, f"N{k}", mode)
            if os.path.isdir(mode_dir):
                for qid in os.listdir(mode_dir):
                    if os.path.isdir(os.path.join(mode_dir, qid)):
                        qids.add(qid)
        qids = sorted(qids, key=lambda x: (len(x), x))

        for qid in qids:
            for i in range(num_inputs):
                f5  = in_path(5,  mode, qid, i)
                f10 = in_path(10, mode, qid, i)
                f20 = in_path(20, mode, qid, i)

                # need all three to test the full chain
                if not (os.path.exists(f5) and os.path.exists(f10) and os.path.exists(f20)):
                    continue

                j5  = _safe_load(f5)
                j10 = _safe_load(f10)
                j20 = _safe_load(f20)
                if not (j5 and j10 and j20):
                    continue

                in5  = j5.get("input_docids", [])
                in10 = j10.get("input_docids", [])
                in20 = j20.get("input_docids", [])

                # ensure sizes are correct (optional, but keeps things clean)
                if not (len(in5)==5 and len(in10)==10 and len(in20)==20):
                    continue

                set5, set10, set20 = set(in5), set(in10), set(in20)
                if set5.issubset(set10) and set10.issubset(set20):
                    print(f"{dataset_short}\t{model_short}\t{mode}\tQID={qid}\tInput={i:02d}  ✔ N5⊆N10⊆N20")
                    printed += 1
                    per_mode_counts[mode] += 1

    # tiny summary
    if printed == 0:
        print("No (mode, qid, input) triplets satisfied N5⊆N10⊆N20.")
    else:
        print("\n— Summary (only satisfied cases) —")
        for m, cnt in per_mode_counts.items():
            print(f"{m}: {cnt} cases")
        print(f"Total: {printed} cases")


# print_subset_chain_ok(
#     dataset_short="trec-covid",
#     model_short="gpt-4o-mini",
#     modes=("single_nonrelevant","half_relevant","single_relevant"),
#     num_inputs=50
# )

# print_subset_chain_ok(
#     dataset_short="trec-covid",
#     model_short="gpt-3.5",
#     modes=("single_nonrelevant","half_relevant","single_relevant"),
#     num_inputs=50
# )

# print_subset_chain_ok(
#     dataset_short="nfcorpus",
#     model_short="gpt-4o-mini",
#     modes=("single_nonrelevant","half_relevant","single_relevant"),
#     num_inputs=50
# )


# print_subset_chain_ok(
#     dataset_short="nfcorpus",
#     model_short="gpt-3.5",
#     modes=("single_nonrelevant","half_relevant","single_relevant"),
#     num_inputs=50
# )

############################################################################################################################
##################################### generate the plot for comparing the size #############################################
############################################################################################################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot size vs. consistency and effectiveness from RankGPT-style logs.

Creates, per (dataset, model):
  figs/{dataset}_{model}_consistency.pdf   # 6 subplots (τ & RBO per mode)
  figs/{dataset}_{model}_ndcg.pdf          # 3 subplots (nDCG per mode)

Assumes log layout:
  results/{dataset}_{model}/N{5,10,20}/{mode}/{qid}/inXX_run{0,1}.json
Where each JSON has keys: input_docids, output_docids, qid, size, mode, ...

Modes expected: ["half_relevant","single_relevant","single_nonrelevant"]
"""

import os, json, math, numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
from tqdm import tqdm
from pyserini.search import get_qrels
from rbo import RankingSimilarity

# ----------------------- config -----------------------
# DATASETS = ["trec-covid", "nfcorpus"]
# MODELS   = ["gpt-3.5", "gpt-4o-mini"]
DATASETS = ["trec-covid"]
MODELS   = ["gpt-3.5"]
MODES    = ["single_relevant", "half_relevant", "single_nonrelevant"]
SIZES    = [5, 10, 20]
NUM_INPUTS = 50
RESULTS_DIR = "results"
FIG_DIR = "figs"
os.makedirs(FIG_DIR, exist_ok=True)

# RBO p for top-k lists (matches your earlier use)
RBO_P = {5: 0.63, 10: 0.79, 20: 0.89}

# ----------------------- helpers -----------------------
def _load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def _topics_name(dataset_short):
    return "beir-v1.0.0-trec-covid-test" if dataset_short == "trec-covid" else "beir-v1.0.0-nfcorpus-test"

def _robust_qrels(qrels_raw, qid):
    # accept str/int QIDs
    if qid in qrels_raw: return qrels_raw[qid]
    try:
        qi = int(qid)
        if qi in qrels_raw: return qrels_raw[qi]
    except Exception:
        pass
    qs = str(qid)
    return qrels_raw.get(qs, {})

def dcg_at_k(labels, k):
    return sum(lab / math.log2(i+2) for i, lab in enumerate(labels[:k]))

def ndcg_for_out(out_docids, qrel_dict, k):
    labs = [int(qrel_dict.get(d, 0)) for d in out_docids[:k]]
    dcg  = dcg_at_k(labs, k)
    ideal = sorted(labs, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return (dcg / idcg) if idcg > 0 else 0.0

def rbo_score(a_ids, b_ids, p):
    # guard duplicates: RBO implementation asserts unique elements
    a = list(OrderedDict.fromkeys(a_ids))
    b = list(OrderedDict.fromkeys(b_ids))
    return RankingSimilarity(a, b).rbo(p=p)

def load_input_ids(root, size, mode, qid, inp_idx):
    f = os.path.join(root, f"N{size}", mode, str(qid), f"in{inp_idx:02d}_run0.json")
    j = _load_json(f)
    return j.get("input_docids", []) if j else None

def load_output_ids(root, size, mode, qid, inp_idx, run):
    f = os.path.join(root, f"N{size}", mode, str(qid), f"in{inp_idx:02d}_run{run}.json")
    j = _load_json(f)
    return j.get("output_docids", []) if j else None

def per_qid_metrics(dataset_short, model_short):
    """
    Returns:
      tau_map[mode][k][qid]  = average τ across inputs (on shared@k), ignoring NaNs
      rbo_map[mode][k][qid]  = average RBO across inputs (on shared@k)
      ndcg_map[mode][k][qid] = average nDCG across inputs & runs
    """
    topics = _topics_name(dataset_short)
    qrels  = get_qrels(topics)
    root   = os.path.join(RESULTS_DIR, f"{dataset_short}_{model_short}")

    tau_map  = {m: {k: {} for k in SIZES} for m in MODES}
    rbo_map  = {m: {k: {} for k in SIZES} for m in MODES}
    ndcg_map = {m: {k: {} for k in SIZES} for m in MODES}

    if not os.path.isdir(root):
        print(f"⚠️  Missing {root}")
        return tau_map, rbo_map, ndcg_map

    for mode in MODES:
        # collect qids present in any size
        qid_set = set()
        for k in SIZES:
            mode_dir = os.path.join(root, f"N{k}", mode)
            if os.path.isdir(mode_dir):
                for qid in os.listdir(mode_dir):
                    if os.path.isdir(os.path.join(mode_dir, qid)):
                        qid_set.add(qid)

        qids = sorted(qid_set, key=lambda x: (len(x), x))

        for qid in tqdm(qids, desc=f"{dataset_short}_{model_short} {mode}"):
            qrel_dict = _robust_qrels(qrels, qid)

            for k in SIZES:
                # we will average over inputs
                taus, rbos, ndcgs = [], [], []

                for i in range(NUM_INPUTS):
                    # inputs for shared computation
                    in5  = load_input_ids(root, 5,  mode, qid, i)
                    in10 = load_input_ids(root, 10, mode, qid, i)
                    in20 = load_input_ids(root, 20, mode, qid, i)

                    shared5  = set(in5).intersection(in10, in20) if (in5 and in10 and in20) else set()
                    shared10 = set(in10).intersection(in20)      if (in10 and in20)             else set()
                    shared20 = set(in20)                          if  in20                       else set()

                    out0 = load_output_ids(root, k, mode, qid, i, 0)
                    out1 = load_output_ids(root, k, mode, qid, i, 1)
                    if not out0 or not out1:
                        continue
                    # keep only shared IDs for τ/RBO
                    if k == 5:   shared = shared5
                    elif k == 10: shared = shared10
                    else:         shared = shared20

                    # τ needs ≥2 shared items
                    if shared and len(shared) >= 2:
                        # represent orders as positions
                        a = [d for d in out0 if d in shared]
                        b = [d for d in out1 if d in shared]
                        # Kendall's τ on index sequences
                        pos = {d:i for i,d in enumerate(a)}
                        seq_a = [pos[d] for d in b if d in pos]
                        if len(seq_a) >= 2:
                            # using scipy for τ
                            from scipy.stats import kendalltau
                            tau_val, _ = kendalltau(seq_a, list(range(len(seq_a))))
                            if not np.isnan(tau_val): taus.append(tau_val)
                        # RBO@p
                        rbos.append(rbo_score(a, b, RBO_P[k]))
                    # nDCG@k on full outputs
                    nd0 = ndcg_for_out(out0, qrel_dict, k)
                    nd1 = ndcg_for_out(out1, qrel_dict, k)
                    ndcgs.append(0.5*(nd0+nd1))

                if taus:  tau_map[mode][k][qid]  = float(np.mean(taus))
                if rbos:  rbo_map[mode][k][qid]  = float(np.mean(rbos))
                if ndcgs: ndcg_map[mode][k][qid] = float(np.mean(ndcgs))

    return tau_map, rbo_map, ndcg_map

from matplotlib.lines import Line2D



def per_qid_metrics_shared5(dataset_short, model_short):
    """
    τ and RBO are computed for N5/N10/N20 *all using the same shared@5 IDs*:
      shared5 = intersection( input_docids@N5, input_docids@N10, input_docids@N20 )
    RBO uses p set for top-5 lists (RBO_P[5]).
    nDCG@k remains computed on the full k-length output lists.

    Returns maps identical in shape to per_qid_metrics(...):
      tau_map[mode][k][qid], rbo_map[mode][k][qid], ndcg_map[mode][k][qid]
    """
    topics = _topics_name(dataset_short)
    qrels  = get_qrels(topics)
    root   = os.path.join(RESULTS_DIR, f"{dataset_short}_{model_short}")

    tau_map  = {m: {k: {} for k in SIZES} for m in MODES}
    rbo_map  = {m: {k: {} for k in SIZES} for m in MODES}
    ndcg_map = {m: {k: {} for k in SIZES} for m in MODES}

    if not os.path.isdir(root):
        print(f"⚠️  Missing {root}")
        return tau_map, rbo_map, ndcg_map

    for mode in MODES:
        # collect qids present in any size
        qid_set = set()
        for k in SIZES:
            mode_dir = os.path.join(root, f"N{k}", mode)
            if os.path.isdir(mode_dir):
                for qid in os.listdir(mode_dir):
                    if os.path.isdir(os.path.join(mode_dir, qid)):
                        qid_set.add(qid)

        qids = sorted(qid_set, key=lambda x: (len(x), x))

        for qid in tqdm(qids, desc=f"{dataset_short}_{model_short} {mode} (shared@5)"):
            qrel_dict = _robust_qrels(qrels, qid)

            for k in SIZES:
                taus, rbos, ndcgs = [], [], []

                for i in range(NUM_INPUTS):
                    # Load inputs to build shared@5
                    in5  = load_input_ids(root, 5,  mode, qid, i)
                    in10 = load_input_ids(root, 10, mode, qid, i)
                    in20 = load_input_ids(root, 20, mode, qid, i)

                    if not (in5 and in10 and in20):
                        # need all three to define shared@5 robustly
                        continue
                    shared5 = set(in5).intersection(in10, in20)
                    if len(shared5) < 2:
                        # τ needs at least 2 items
                        continue

                    # outputs at this size (re-ranks to compare)
                    out0 = load_output_ids(root, k, mode, qid, i, 0)
                    out1 = load_output_ids(root, k, mode, qid, i, 1)
                    if not out0 or not out1:
                        continue

                    # filter outputs to shared@5 (preserve order)
                    a = [d for d in out0 if d in shared5]
                    b = [d for d in out1 if d in shared5]

                    # Kendall's τ via relative positions (needs ≥2)
                    if len(a) >= 2 and len(b) >= 2:
                        from scipy.stats import kendalltau
                        pos_a = {d: idx for idx, d in enumerate(a)}
                        seq_a = [pos_a[d] for d in b if d in pos_a]
                        if len(seq_a) >= 2:
                            tau_val, _ = kendalltau(seq_a, list(range(len(seq_a))))
                            if not np.isnan(tau_val):
                                taus.append(float(tau_val))

                        # RBO with p tuned for top-5 lists
                        # (guard duplicates for the rbo lib)
                        a_unique = list(OrderedDict.fromkeys(a))
                        b_unique = list(OrderedDict.fromkeys(b))
                        rbos.append(RankingSimilarity(a_unique, b_unique).rbo(p=RBO_P[5]))

                    # nDCG@k on full lists (unchanged)
                    nd0 = ndcg_for_out(out0, qrel_dict, k)
                    nd1 = ndcg_for_out(out1, qrel_dict, k)
                    ndcgs.append(0.5 * (nd0 + nd1))

                if taus:  tau_map[mode][k][qid]  = float(np.mean(taus))
                if rbos:  rbo_map[mode][k][qid]  = float(np.mean(rbos))
                if ndcgs: ndcg_map[mode][k][qid] = float(np.mean(ndcgs))

    return tau_map, rbo_map, ndcg_map




PASTEL = {"N5":"#C7EA46", "N10":"#F6BDC7" , "N20":"#C1D9E9" }
DARK   = {"N5":"#6E8B3D", "N10":"#A33A4B" , "N20":"#3B688A"}
LINEWIDTH  = 2.8
MARKERSIZE = 6.0


# ======= style controls =======
AXIS_LABEL_FONTSIZE = 17
TICK_FONTSIZE       = 17
TITLE_FONTSIZE      = 17

LEGEND_TITLE        = "List Size"
LEGEND_FONTSIZE     = 18          # labels
LEGEND_TITLE_FONTSIZE = 17
LEGEND_HANDLELEN    = 2.0
LEGEND_MARKERSIZE   = 10.0
LEGEND_LINEWIDTH    = 7.5

# Position the shared legend (relative to the whole figure)
# Move these to snug the legend closer/farther from the last subplot.
# LEGEND_BBOX   = (0.5, -0.0005)
FIG_BOTTOM = 0.10   # increase if legend overlaps
FIG_LEFT   = 0.08
FIG_RIGHT  = 0.98
FIG_HSPACE = 0.32   # vertical spacing between rows


def _plot_lines_and_points(ax, order_qids, series_by_k, ylabel, title, ylim, yticks):
    from matplotlib.ticker import FormatStrFormatter
    for k, y in series_by_k.items():
        # pastel line
        ax.plot(range(len(order_qids)), y,
                linewidth=LINEWIDTH, marker=None, color=PASTEL[f"N{k}"])
        # dark dots
        ax.plot(range(len(order_qids)), y,
                linestyle="None", marker='o', markersize=MARKERSIZE, color=DARK[f"N{k}"])

    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.grid(True, alpha=0.3)

    # hide x tick labels by default (we’ll turn on only the last one later)
    ax.set_xticks([])
    ax.set_xlabel("")



from matplotlib.lines import Line2D

def _order_qids_by_k5(metric_map_for_mode):
    if 5 in metric_map_for_mode and metric_map_for_mode[5]:
        k5 = metric_map_for_mode[5]
        return sorted(k5.keys(), key=lambda q: k5[q])  # ascending
    all_qids = set().union(*[
        set(metric_map_for_mode[k].keys())
        for k in metric_map_for_mode if metric_map_for_mode[k]
    ])
    return sorted(all_qids, key=lambda x: (len(x), x))

def _plot_consistency(dataset_short, model_short, tau_map, rbo_map):
    from matplotlib.ticker import FormatStrFormatter
    fig, axes = plt.subplots(6, 1, figsize=(12, 16))
    rowspec = []
    for mode in MODES:
        rowspec.append((mode, "tau"))
        rowspec.append((mode, "rbo"))

    for row_idx, (mode, which) in enumerate(rowspec):
        ax = axes[row_idx]
        metric_map_for_mode = tau_map[mode] if which == "tau" else rbo_map[mode]
        order_qids = _order_qids_by_k5(metric_map_for_mode)

        series_by_k = {}
        if which == "tau":
            ylabel = "TAU"
            ylim   = (0.65, 1.00)
            yticks = [0.75, 0.85, 0.95]
            title  = f"{mode} — Kendall’s τ"
        else:
            ylabel = "RBO"
            ylim   = (0.70, 0.90)   # your requested cap at 0.90
            yticks = [0.75, 0.85, 0.95]
            title  = f"{mode} — RBO@p"

        for k in SIZES:
            y = ( [tau_map[mode][k].get(q, np.nan) for q in order_qids]
                  if which == "tau"
                  else [rbo_map[mode][k].get(q, np.nan) for q in order_qids] )
            series_by_k[k] = y

        _plot_lines_and_points(ax, order_qids, series_by_k, ylabel, title, ylim, yticks)

    # Only the last subplot shows the x label
    axes[-1].set_xlabel("Query (sorted by k' = 5 score)", fontsize=AXIS_LABEL_FONTSIZE)
    axes[-1].tick_params(axis='x', labelsize=TICK_FONTSIZE)

    # Position legend and tighten layout
#     _make_shared_legend(fig)

    last_ax = axes[-1]
    handles = []
    for k in [5,10,20]:
        handles.append(Line2D([0], [0],
                              color=PASTEL[f"N{k}"],
                              marker='o', markerfacecolor=DARK[f"N{k}"],
                              markeredgecolor=DARK[f"N{k}"],
                              linewidth=LEGEND_LINEWIDTH, markersize=LEGEND_MARKERSIZE,
                              label=f"k'={k}"))

    fig.legend(handles=handles,
               title=LEGEND_TITLE,
               loc="lower center",
               ncol=3,
               frameon=False,
               fontsize=LEGEND_FONTSIZE,
               title_fontsize=LEGEND_TITLE_FONTSIZE,
               handlelength=LEGEND_HANDLELEN,
               borderaxespad=0.0,
               # Anchor relative to the LAST AXES, not the whole figure:
               bbox_to_anchor=(0.5, -0.75),          # ← make this less negative to move closer
               bbox_transform=last_ax.transAxes)
    
    fig.subplots_adjust(bottom=FIG_BOTTOM, left=FIG_LEFT, right=FIG_RIGHT, hspace=FIG_HSPACE)

    out = os.path.join(FIG_DIR, f"{dataset_short}_{model_short}_size_consistency_5.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"✓ wrote {out}")


    out = os.path.join(FIG_DIR, f"{dataset_short}_{model_short}_size_consistency_5.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"✓ wrote {out}")


def _plot_ndcg(dataset_short, model_short, ndcg_map):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for row, mode in enumerate(MODES):
        ax = axes[row]
        order_qids = _order_qids_by_k5(ndcg_map[mode])
        series_by_k = {k: [ndcg_map[mode][k].get(q, np.nan) for q in order_qids] for k in SIZES}
        _plot_lines_and_points(ax, order_qids, series_by_k,
                               ylabel="nDCG@k", title=f"{mode} — nDCG@k",
                               ylim=(0.0, 1.05), yticks=None)

    axes[-1].set_xlabel("Query (sorted by k' = 5 score)", fontsize=AXIS_LABEL_FONTSIZE)
    axes[-1].tick_params(axis='x', labelsize=TICK_FONTSIZE)

#     _make_shared_legend(fig)

    last_ax = axes[-1]
    handles = []
    for k in [5,10,20]:
        handles.append(Line2D([0], [0],
                              color=PASTEL[f"N{k}"],
                              marker='o', markerfacecolor=DARK[f"N{k}"],
                              markeredgecolor=DARK[f"N{k}"],
                              linewidth=LEGEND_LINEWIDTH, markersize=LEGEND_MARKERSIZE,
                              label=f"k'={k}"))

    fig.legend(handles=handles,
               title=LEGEND_TITLE,
               loc="lower center",
               ncol=3,
               frameon=False,
               fontsize=LEGEND_FONTSIZE,
               title_fontsize=LEGEND_TITLE_FONTSIZE,
               handlelength=LEGEND_HANDLELEN,
               borderaxespad=0.0,
               # Anchor relative to the LAST AXES, not the whole figure:
               bbox_to_anchor=(0.5, -0.45),          # ← make this less negative to move closer
               bbox_transform=last_ax.transAxes)
    
    fig.subplots_adjust(bottom=FIG_BOTTOM, left=FIG_LEFT, right=FIG_RIGHT, hspace=FIG_HSPACE)

    out = os.path.join(FIG_DIR, f"{dataset_short}_{model_short}_size_ndcg.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"✓ wrote {out}")





# ----------------------- main -----------------------

def main_plot():
    for dataset_short in DATASETS:
        for model_short in MODELS:
            print(f"\n=== {dataset_short} × {model_short} (τ/RBO on shared@5) ===")
            # Use the shared@5 metrics for τ/RBO (nDCG unchanged)
            tau_map, rbo_map, ndcg_map = per_qid_metrics_shared5(dataset_short, model_short)
            _plot_consistency(dataset_short, model_short, tau_map, rbo_map)
            _plot_ndcg(dataset_short, model_short, ndcg_map)

# main_plot()



############################################################################################################################
############################################# same as above plot but only has rbo #######################
############################################################################################################################

# ===================== RBO-ONLY PLOTTING (3 separate figures) =====================

# ===================== RBO-ONLY (single PDF with 3 subplots) =====================

def _plot_rbo_only_single_pdf(dataset_short, model_short, rbo_map):
    """
    Generates a SINGLE PDF containing all three RBO subplots (one per mode).
    Saves:
      figs/{dataset}_{model}_rbo.pdf
    """
    from matplotlib.ticker import FormatStrFormatter
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(3, 1, figsize=(9, 7))

    for row, mode in enumerate(MODES):
        ax = axes[row]

        # Same ordering scheme as your other plots: sort by k'=5 (ascending)
        order_qids = _order_qids_by_k5(rbo_map[mode])

        series_by_k = {
            k: [rbo_map[mode][k].get(q, np.nan) for q in order_qids]
            for k in SIZES
        }

        _plot_lines_and_points(
            ax,
            order_qids,
            series_by_k,
            ylabel="RBO",
            title=f"{mode} — RBO@p",
            ylim=(0.70, 0.90),
            yticks=[0.75, 0.85, 0.95],
        )

        # Keep y tick formatting consistent
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Only bottom subplot shows x-axis label (like your other multi-row plots)
    axes[-1].set_xlabel("Query (sorted by k' = 5 score)", fontsize=AXIS_LABEL_FONTSIZE)
    axes[-1].tick_params(axis='x', labelsize=TICK_FONTSIZE)

    # Shared legend at the bottom
    last_ax = axes[-1]
    handles = []
    for k in [5, 10, 20]:
        handles.append(Line2D(
            [0], [0],
            color=PASTEL[f"N{k}"],
            marker='o',
            markerfacecolor=DARK[f"N{k}"],
            markeredgecolor=DARK[f"N{k}"],
            linewidth=LEGEND_LINEWIDTH,
            markersize=LEGEND_MARKERSIZE,
            label=f"k'={k}"
        ))

    fig.legend(
        handles=handles,
        title=LEGEND_TITLE,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
        handlelength=LEGEND_HANDLELEN,
        borderaxespad=0.0,
        # Anchor relative to last axis (same style you used)
        bbox_to_anchor=(0.5, -0.70),
        bbox_transform=last_ax.transAxes
    )

    fig.subplots_adjust(bottom=FIG_BOTTOM, left=FIG_LEFT, right=FIG_RIGHT, hspace=FIG_HSPACE)

    out = os.path.join(FIG_DIR, f"{dataset_short}_{model_short}_rbo_size_consistency_5.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ wrote {out}")


def main_plot_rbo_only_single_pdf():
    for dataset_short in DATASETS:
        for model_short in MODELS:
            print(f"\n=== {dataset_short} × {model_short} (RBO-only single PDF; shared@5) ===")
            _, rbo_map, _ = per_qid_metrics_shared5(dataset_short, model_short)
            _plot_rbo_only_single_pdf(dataset_short, model_short, rbo_map)

# Uncomment to run:
main_plot_rbo_only_single_pdf()


############################################################################################################################
############################################# fit the line to see the correlation between the 3 sizes #######################
############################################################################################################################

from scipy.stats import linregress
from scipy.stats import linregress
from matplotlib.ticker import FormatStrFormatter


# One color per column (k-pair)
COLORS_PER_COLUMN = {
    0: "purple",      # col 1: k=5 vs k=10
    1: "darkorange",  # col 2: k=5 vs k=20
    2: "teal",        # col 3: k=10 vs k=20
}


PAIRWISE_PAIRS = [(5, 10), (5, 20), (10, 20)]  # (k1, k2) per column


PAIRWISE_PAIRS = [(5, 10), (5, 20), (10, 20)]  # (k1, k2) per column


def _pairwise_scatter_for_mode(ax, metric_map, mode, k1, k2, metric_label, col_idx):
    """
    metric_map: tau_map or rbo_map
      metric_map[mode][k][qid] = value

    Plots scatter for (k1 vs k2) for a given mode, fits regression line,
    and annotates r and p.

    col_idx: 0,1,2 → chooses color from COLORS_PER_COLUMN
    """
    m1 = metric_map[mode][k1]
    m2 = metric_map[mode][k2]

    # qids that have both k1 and k2
    qids = sorted(set(m1.keys()) & set(m2.keys()), key=lambda x: (len(x), x))

    xs, ys = [], []
    for q in qids:
        v1 = m1.get(q, np.nan)
        v2 = m2.get(q, np.nan)
        if np.isnan(v1) or np.isnan(v2):
            continue
        xs.append(v1)
        ys.append(v2)

    if len(xs) >= 2:
        xs = np.array(xs)
        ys = np.array(ys)

        # Fit regression: y = intercept + slope * x
        res = linregress(xs, ys)
        slope, intercept, r_value, p_value = res.slope, res.intercept, res.rvalue, res.pvalue

        color = COLORS_PER_COLUMN[col_idx]

        # Scatter points
        ax.scatter(xs, ys, alpha=0.8, s=35, color=color)

        # Common axis limits so diagonal is 45°
        vmin = float(min(xs.min(), ys.min()))
        vmax = float(max(xs.max(), ys.max()))
        if vmax == vmin:
            vmin -= 0.1
            vmax += 0.1
        margin = 0.05 * (vmax - vmin)
        lo = vmin - margin
        hi = vmax + margin

        # Diagonal y = x
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, color="gray")

        # Regression line
        x_line = np.linspace(lo, hi, 100)
        y_line = intercept + slope * x_line
        ax.plot(x_line, y_line, linewidth=2.5, color=color)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        # 2-decimal ticks
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # Bigger r / p annotation
        text = f"r={r_value:.2f}\np={p_value:.2g}"
        ax.text(
            0.05, 0.95, text,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=15,              # bigger
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", alpha=0.8,
                      edgecolor="none"),
        )
    else:
        # Not enough points to fit anything
        ax.text(
            0.5, 0.5, "n < 2",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=12,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)


def _plot_pairwise_scatter_all(metric_map, metric_label, dataset_short, model_short):
    """
    metric_map: tau_map or rbo_map
    metric_label: "tau" or "rbo" (only used in filename).
    Creates a 3 (modes) x 3 (pairs) grid of scatter+regression plots.
    """
    num_modes = len(MODES)
    fig, axes = plt.subplots(num_modes, 3, figsize=(15, 13))

    # If only one mode, axes may be 1D; normalize to 2D
    if num_modes == 1:
        axes = np.array([axes])

    # Plot all subplots
    for row, mode in enumerate(MODES):
        for col, (k1, k2) in enumerate(PAIRWISE_PAIRS):
            ax = axes[row, col]
            _pairwise_scatter_for_mode(
                ax, metric_map, mode, k1, k2, metric_label, col_idx=col
            )

            # Axis labels as "k=5", "k=10", ...
            ax.set_xlabel(f"k={k1}", fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_ylabel(f"k={k2}", fontsize=AXIS_LABEL_FONTSIZE)

    # Remove overall title (you don't want it)
    # fig.suptitle(...)

    # More vertical space between rows, and a bit more left margin
    fig.subplots_adjust(
        left=0.12,   # more space on the left for row labels
        right=0.98,
        top=0.98,
        bottom=0.08,
        hspace=0.30,  # more vertical spacing between rows
        wspace=0.35,  # some horizontal spacing too
    )

    # Add row labels (Single rel / Half rel / Single nonrel) to the left,
    # with a bit of padding from the axes.
    for row, mode in enumerate(MODES):
        # get vertical middle of this row of axes
        row_axes = axes[row, :]
        y_mid = 0.5 * (row_axes[0].get_position().y0 + row_axes[0].get_position().y1)
        row_label = mode.replace("_", " ")  # e.g. "single relevant"
        fig.text(
            0.03, y_mid, row_label,
            va="center", ha="center",
            fontsize=AXIS_LABEL_FONTSIZE + 1, rotation=90
        )

    out = os.path.join(
        FIG_DIR,
        f"{dataset_short}_{model_short}_{metric_label}_pairwise_scatter_shared5.pdf"
    )
    fig.savefig(out, bbox_inches="tight")
    print(f"✓ wrote {out}")


def main_plot_correlation():
    for dataset_short in DATASETS:
        for model_short in MODELS:
            print(f"\n=== {dataset_short} × {model_short} (τ/RBO on shared@5) ===")
            tau_map, rbo_map, ndcg_map = per_qid_metrics_shared5(dataset_short, model_short)

            # NEW pairwise scatter plots
            _plot_pairwise_scatter_all(tau_map, "tau", dataset_short, model_short)
            _plot_pairwise_scatter_all(rbo_map, "rbo", dataset_short, model_short)   
            
# main_plot_correlation()   

##############################################################################################################################
################################# same plot for shared 10 docs ##############################################################
#############################################################################################################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math, numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib.lines import Line2D
from tqdm import tqdm
from pyserini.search import get_qrels
from rbo import RankingSimilarity
from scipy.stats import kendalltau

# ===================== CONFIG =====================
DATASETS = ["trec-covid", "nfcorpus"]
MODELS   = ["gpt-3.5", "gpt-4o-mini"]
MODES    = ["half_relevant", "single_relevant", "single_nonrelevant"]
SIZES    = [10, 20]                 # only 10 and 20
NUM_INPUTS   = 50
RESULTS_DIR  = "results"
FIG_DIR      = "figs"
os.makedirs(FIG_DIR, exist_ok=True)




# Aesthetics
PASTEL = {"N10":"#F6BDC7", "N20":"#C1D9E9"}
DARK   = {"N10":"#A33A4B", "N20":"#3B688A"}
LINEWIDTH = 2.5
MARKERSIZE = 6.5

# Axis/legend sizing & placement
AXIS_LABEL_FONTSIZE = 13
TICK_FONTSIZE       = 12
TITLE_FONTSIZE      = 13

LEGEND_TITLE          = "List Size"
LEGEND_FONTSIZE       = 16
LEGEND_TITLE_FONTSIZE = 14
LEGEND_HANDLELEN      = 2.2
LEGEND_MARKERSIZE     = 10.0
LEGEND_LINEWIDTH      = 7.0

# Make figure box slightly tighter so legend can sit closer
FIG_LEFT   = 0.08
FIG_RIGHT  = 0.98
FIG_BOTTOM = 0.06    # closer to the bottom than before
FIG_TOP    = 0.98
FIG_HSPACE = 0.25

# RBO parameter: use p = 0.79 for shared@10 comparisons
RBO_P_SHARED10 = 0.79
# ==================================================

def _load_json(p):
    try:
        with open(p, "r") as f: return json.load(f)
    except Exception:
        return None

def _topics_name(dshort):
    return "beir-v1.0.0-trec-covid-test" if dshort == "trec-covid" else "beir-v1.0.0-nfcorpus-test"

def _robust_qrels(qrels_raw, qid):
    if qid in qrels_raw: return qrels_raw[qid]
    try:
        qi = int(qid)
        if qi in qrels_raw: return qrels_raw[qi]
    except Exception:
        pass
    return qrels_raw.get(str(qid), {})

def dcg_at_k(labels, k):
    return sum(l / math.log2(i+2) for i, l in enumerate(labels[:k]))

def ndcg_for_out(out_docids, qrel_dict, k):
    labs  = [int(qrel_dict.get(d, 0)) for d in out_docids[:k]]
    dcg   = dcg_at_k(labs, k)
    idcg  = dcg_at_k(sorted(labs, reverse=True), k)
    return (dcg / idcg) if idcg > 0 else 0.0

def rbo_score(a_ids, b_ids, p):
    # guard duplicates; rbo lib asserts uniqueness
    a = list(OrderedDict.fromkeys(a_ids))
    b = list(OrderedDict.fromkeys(b_ids))
    return RankingSimilarity(a, b).rbo(p=p)

def load_input_ids(root, size, mode, qid, inp_idx):
    f = os.path.join(root, f"N{size}", mode, str(qid), f"in{inp_idx:02d}_run0.json")
    j = _load_json(f)
    return j.get("input_docids", []) if j else None

def load_output_ids(root, size, mode, qid, inp_idx, run):
    f = os.path.join(root, f"N{size}", mode, str(qid), f"in{inp_idx:02d}_run{run}.json")
    j = _load_json(f)
    return j.get("output_docids", []) if j else None

def per_qid_metrics_shared10(dataset_short, model_short):
    """
    Compute τ/RBO over the docids shared between N10 and N20 (shared@10).
    For both k=10 and k=20, we restrict the comparison to shared@10.
    nDCG@k uses the full outputs at that k (as usual).
    Returns:
      tau_map[mode][k][qid]  (k in {10,20})
      rbo_map[mode][k][qid]
      ndcg_map[mode][k][qid]
    """
    topics = _topics_name(dataset_short)
    qrels  = get_qrels(topics)
    root   = os.path.join(RESULTS_DIR, f"{dataset_short}_{model_short}")

    tau_map  = {m: {k: {} for k in SIZES} for m in MODES}
    rbo_map  = {m: {k: {} for k in SIZES} for m in MODES}
    ndcg_map = {m: {k: {} for k in SIZES} for m in MODES}

    if not os.path.isdir(root):
        print(f"⚠️  Missing {root}")
        return tau_map, rbo_map, ndcg_map

    for mode in MODES:
        # collect qids present in either N10 or N20
        qid_set = set()
        for k in SIZES:
            mode_dir = os.path.join(root, f"N{k}", mode)
            if os.path.isdir(mode_dir):
                for qid in os.listdir(mode_dir):
                    if os.path.isdir(os.path.join(mode_dir, qid)):
                        qid_set.add(qid)
        qids = sorted(qid_set, key=lambda x: (len(x), x))

        for qid in tqdm(qids, desc=f"{dataset_short}_{model_short} {mode} (shared@10)"):
            qrel_dict = _robust_qrels(qrels, qid)

            # average per input
            for k in SIZES:  # k in {10,20}
                taus, rbos, ndcgs = [], [], []
                for i in range(NUM_INPUTS):
                    in10 = load_input_ids(root, 10, mode, qid, i)
                    in20 = load_input_ids(root, 20, mode, qid, i)
                    if not (in10 and in20): 
                        continue
                    shared10 = set(in10).intersection(in20)
                    out0 = load_output_ids(root, k, mode, qid, i, 0)
                    out1 = load_output_ids(root, k, mode, qid, i, 1)
                    if not (out0 and out1):
                        continue

                    # restrict to shared@10 for τ/RBO
                    if len(shared10) >= 2:
                        a = [d for d in out0 if d in shared10]
                        b = [d for d in out1 if d in shared10]
                        # τ on index sequences
                        pos = {d: idx for idx, d in enumerate(a)}
                        seq_a = [pos[d] for d in b if d in pos]
                        if len(seq_a) >= 2:
                            tval, _ = kendalltau(seq_a, list(range(len(seq_a))))
                            if not np.isnan(tval):
                                taus.append(tval)
                        rbos.append(rbo_score(a, b, RBO_P_SHARED10))

                    # nDCG@k (full outputs)
                    nd0 = ndcg_for_out(out0, qrel_dict, k)
                    nd1 = ndcg_for_out(out1, qrel_dict, k)
                    ndcgs.append(0.5*(nd0+nd1))

                if taus:  tau_map[mode][k][qid]  = float(np.mean(taus))
                if rbos:  rbo_map[mode][k][qid]  = float(np.mean(rbos))
                if ndcgs: ndcg_map[mode][k][qid] = float(np.mean(ndcgs))

    return tau_map, rbo_map, ndcg_map

def _order_qids_by_k10(metric_map_for_mode):
    """Sort QIDs ascending by the k′=10 values of the supplied metric map."""
    if 10 in metric_map_for_mode and metric_map_for_mode[10]:
        k10 = metric_map_for_mode[10]
        return sorted(k10.keys(), key=lambda q: k10[q])  # ascending
    # fallback to union
    all_qids = set().union(*[set(metric_map_for_mode[k].keys())
                             for k in metric_map_for_mode if metric_map_for_mode[k]])
    return sorted(all_qids, key=lambda x: (len(x), x))

def _plot_lines_and_points(ax, order_qids, series_by_k, ylabel, title, ylim, yticks):
    for k, y in series_by_k.items():
        ax.plot(range(len(order_qids)), y, linewidth=LINEWIDTH, color=PASTEL[f"N{k}"])
        ax.plot(range(len(order_qids)), y, linestyle="None", marker='o',
                markersize=MARKERSIZE, color=DARK[f"N{k}"])
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:.2f}" for t in yticks], fontsize=TICK_FONTSIZE)
    # hide x ticks (we’ll only label the last subplot)
    ax.set_xticks([])
    ax.grid(True, alpha=0.3)

def _make_shared_legend(fig):
    handles = []
    for k in [10, 20]:
        handles.append(Line2D([0], [0],
                              color=PASTEL[f"N{k}"],
                              marker='o', markerfacecolor=DARK[f"N{k}"],
                              markeredgecolor=DARK[f"N{k}"],
                              linewidth=LEGEND_LINEWIDTH, markersize=LEGEND_MARKERSIZE,
                              label=f"k'={k}"))
    leg = fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
                     title=LEGEND_TITLE, fontsize=LEGEND_FONTSIZE,
                     title_fontsize=LEGEND_TITLE_FONTSIZE, handlelength=LEGEND_HANDLELEN,
                     bbox_to_anchor=(0.5, -0.01))  # closer to last subplot
    return leg

def _plot_consistency_shared10(dataset_short, model_short, tau_map, rbo_map):
    """
    6 subplots in a single column:
      Row1: half_relevant τ (shared@10)
      Row2: half_relevant RBO (shared@10, p=0.79)
      Row3: single_relevant τ
      Row4: single_relevant RBO
      Row5: single_nonrelevant τ
      Row6: single_nonrelevant RBO
    Curves only for N10 and N20.
    """
    fig, axes = plt.subplots(6, 1, figsize=(12, 16))
    rowspec = []
    for mode in MODES:
        rowspec.append((mode, "tau"))
        rowspec.append((mode, "rbo"))

    for row_idx, (mode, which) in enumerate(rowspec):
        ax = axes[row_idx]
        metric_map_for_mode = tau_map[mode] if which == "tau" else rbo_map[mode]
        order_qids = _order_qids_by_k10(metric_map_for_mode)

        if which == "tau":
            ylim   = (0.70, 1.00)
            yticks = [0.70, 0.80, 0.90, 1.00]
            ylabel = "TAU"
            title  = f"{mode} — Kendall’s τ"
        else:
            ylim   = (0.75, 0.90)
            yticks = [0.75, 0.80, 0.85, 0.90]
            ylabel = "RBO"
            title  = f"{mode} — RBO"

        series_by_k = {}
        for k in SIZES:  # 10, 20
            if which == "tau":
                y = [tau_map[mode][k].get(q, np.nan) for q in order_qids]
            else:
                y = [rbo_map[mode][k].get(q, np.nan) for q in order_qids]
            series_by_k[k] = y

        _plot_lines_and_points(ax, order_qids, series_by_k, ylabel, title, ylim, yticks)

    # Add x-label only to the last subplot
    axes[-1].set_xlabel("Query (sorted by k' = 10 score)", fontsize=AXIS_LABEL_FONTSIZE)
    for tick in axes[-1].get_xticklabels():
        tick.set_fontsize(TICK_FONTSIZE)

    # Layout & legend
    fig.subplots_adjust(left=FIG_LEFT, right=FIG_RIGHT, top=FIG_TOP, bottom=FIG_BOTTOM, hspace=FIG_HSPACE)
    _make_shared_legend(fig)
    out = os.path.join(FIG_DIR, f"{dataset_short}_{model_short}_size_consistency_shared10.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"✓ wrote {out}")

def _plot_ndcg_shared10(dataset_short, model_short, ndcg_map):
    """3 subplots (one per mode), curves for N10 and N20, QIDs sorted by nDCG@10."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for row, mode in enumerate(MODES):
        ax = axes[row]
        order_qids = _order_qids_by_k10(ndcg_map[mode])

        series_by_k = {k: [ndcg_map[mode][k].get(q, np.nan) for q in order_qids] for k in SIZES}
        _plot_lines_and_points(
            ax, order_qids, series_by_k,
            ylabel="nDCG@k",
            title=f"{mode} — nDCG@k",
            ylim=(0.0, 1.05),
            yticks=[0.0, 0.25, 0.50, 0.75, 1.00]
        )

    axes[-1].set_xlabel("Query (sorted by k' = 10 score)", fontsize=AXIS_LABEL_FONTSIZE)
    for tick in axes[-1].get_xticklabels():
        tick.set_fontsize(TICK_FONTSIZE)

    fig.subplots_adjust(left=FIG_LEFT, right=FIG_RIGHT, top=FIG_TOP, bottom=FIG_BOTTOM, hspace=FIG_HSPACE)
    _make_shared_legend(fig)
    out = os.path.join(FIG_DIR, f"{dataset_short}_{model_short}_size_ndcg_shared10.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"✓ wrote {out}")

def main_plot_shared10():
    for dataset_short in DATASETS:
        for model_short in MODELS:
            print(f"\n=== {dataset_short} × {model_short} (shared@10, p=0.79) ===")
            tau_map, rbo_map, ndcg_map = per_qid_metrics_shared10(dataset_short, model_short)
            _plot_consistency_shared10(dataset_short, model_short, tau_map, rbo_map)
            _plot_ndcg_shared10(dataset_short, model_short, ndcg_map)

# if __name__ == "__main__":
#     main_plot_shared10()




############################################################################################################################
############################################# fit the line to see the correlation between the 2 sizes #######################
############################################################################################################################

from scipy.stats import linregress
from matplotlib.ticker import FormatStrFormatter


# One solid color per column (per mode) for the scatter/regression plots
SCATTER_COLORS = {
    0: "purple",      # half_relevant
    1: "darkorange",  # single_relevant
    2: "teal",        # single_nonrelevant
}


def _scatter_k10_vs_k20_for_mode(ax, metric_map, mode, col_idx):
    """
    metric_map: tau_map or rbo_map from per_qid_metrics_shared10
      metric_map[mode][k][qid] = value, for k in {10,20}.

    Plots scatter for (k=10 vs k=20) for a given mode, fits regression line,
    and annotates r and p.
    """
    m10 = metric_map[mode][10]
    m20 = metric_map[mode][20]

    # qids that have both k=10 and k=20
    qids = sorted(set(m10.keys()) & set(m20.keys()), key=lambda x: (len(x), x))

    xs, ys = [], []
    for q in qids:
        v10 = m10.get(q, np.nan)
        v20 = m20.get(q, np.nan)
        if np.isnan(v10) or np.isnan(v20):
            continue
        xs.append(v10)
        ys.append(v20)

    if len(xs) >= 2:
        xs = np.array(xs)
        ys = np.array(ys)

        # Fit regression: y = intercept + slope * x
        res = linregress(xs, ys)
        slope, intercept, r_value, p_value = res.slope, res.intercept, res.rvalue, res.pvalue

        color = SCATTER_COLORS[col_idx]

        # Scatter
        ax.scatter(xs, ys, alpha=0.8, s=35, color=color)

        # Common axis limits so diagonal is 45°
        vmin = float(min(xs.min(), ys.min()))
        vmax = float(max(xs.max(), ys.max()))
        if vmax == vmin:
            vmin -= 0.1
            vmax += 0.1
        margin = 0.05 * (vmax - vmin)
        lo = vmin - margin
        hi = vmax + margin

        # Diagonal y = x
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, color="gray")

        # Regression line
        x_line = np.linspace(lo, hi, 100)
        y_line = intercept + slope * x_line
        ax.plot(x_line, y_line, linewidth=2.5, color=color)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        # Two decimal ticks
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # Annotation: r and p (big, not bold)
        text = f"r={r_value:.2f}\np={p_value:.2g}"
        ax.text(
            0.05, 0.95, text,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=15,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                alpha=0.8,
                edgecolor="none",
            ),
        )
    else:
        # Not enough points to fit anything
        ax.text(
            0.5, 0.5, "n < 2",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=12,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)


def _plot_pairwise_k10_vs_k20_shared10(metric_map, metric_label, dataset_short, model_short):
    """
    metric_map: tau_map or rbo_map from per_qid_metrics_shared10
    metric_label: "tau" or "rbo" (used only in filename).

    Creates a single row with 3 columns:
      col 0: half_relevant
      col 1: single_relevant
      col 2: single_nonrelevant
    Each subplot: k=10 vs k=20 scatter + regression + diagonal.
    """
    num_modes = len(MODES)
    assert num_modes == 3, "This layout assumes exactly three MODES."

    fig, axes = plt.subplots(1, num_modes, figsize=(15, 4.5))

    # Ensure axes is 1D array
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for col, mode in enumerate(MODES):
        ax = axes[col]
        _scatter_k10_vs_k20_for_mode(ax, metric_map, mode, col_idx=col)

        # Axis labels
        ax.set_xlabel("k=10", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("k=20", fontsize=AXIS_LABEL_FONTSIZE)

        # Optional: a small title for each mode, or you can omit this
        mode_label = mode.replace("_", " ")
        ax.set_title(mode_label, fontsize=TITLE_FONTSIZE)

    # Layout: more space between columns, slightly less vertical padding
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.95,
        bottom=0.12,
        hspace=0.10,   # only one row, so this is not critical
        wspace=0.40,   # more space between columns
    )

    out = os.path.join(
        FIG_DIR,
        f"{dataset_short}_{model_short}_{metric_label}_k10_vs_k20_shared10_scatter.pdf"
    )
    fig.savefig(out, bbox_inches="tight")
    print(f"✓ wrote {out}")

def main_plot_correlation_shared10():
    for dataset_short in DATASETS:
        for model_short in MODELS:
            print(f"\n=== {dataset_short} × {model_short} (shared@10, p=0.79) ===")
            tau_map, rbo_map, ndcg_map = per_qid_metrics_shared10(dataset_short, model_short)

            # NEW single-row scatter plots: k=10 vs k=20 for each mode
            _plot_pairwise_k10_vs_k20_shared10(tau_map, "tau", dataset_short, model_short)
            _plot_pairwise_k10_vs_k20_shared10(rbo_map, "rbo", dataset_short, model_short)

            
# main_plot_correlation_shared10()