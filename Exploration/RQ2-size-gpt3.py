# reuse_gpt4_inputs_for_gpt35.py
# -----------------------------------------------------------------------------
# Reuse EXACT inputs (qids + input_docids) from an existing GPT-4/4o-mini run
# and re-run them with GPT-3.5-turbo, preserving folder structure:
#
#   SOURCE: results/{DATASET_SHORT}_{SOURCE_MODEL_SHORT}/N{5|10|20}/{mode}/{qid}/inXX_run{0|1}.json
#   TARGET: results/{DATASET_SHORT}_{TARGET_MODEL_SHORT}/N{5|10|20}/{mode}/{qid}/inXX_run{0|1}.json
#
# Before calling the LLM, we:
#   - Verify subset chain N5⊆N10⊆N20 when all present
#   - Verify relevant/non-relevant counts per mode using qrels
#
# Safe to resume: skips inputs already present under TARGET.
# Set CHECK_ONLY=True to only verify parity (no LLM calls).
# -----------------------------------------------------------------------------

import os
import re
import json
import copy
import random
from typing import Dict, List, Tuple, Optional

from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels

# from your project
from rank_gpt import create_permutation_instruction, run_llm, receive_permutation

# ========================
# CONFIGURE HERE
# ========================

# Dataset: "trec-covid" or "nfcorpus"
DATASET_SHORT = "trec-covid"          # ← change if needed: "trec-covid" | "nfcorpus"

# Source (already completed) model short tag in folder names
SOURCE_MODEL_SHORT = "gpt-4o-mini"    # e.g., "gpt-4o-mini" or "gpt-4o" or "gpt-4"

# Target: the model you want to run now
TARGET_MODEL_NAME  = "gpt-3.5-turbo"
TARGET_MODEL_SHORT = "gpt-3.5"

# Modes & Sizes to process (must match what exists in SOURCE)
MODES = ["half_relevant", "single_relevant", "single_nonrelevant"]
SIZES = [5, 10, 20]

# Two deterministic runs per input
DETER_RUNS = 2

# If True: only check parity and print diagnostics; do NOT call the LLM
CHECK_ONLY = False

API_KEY = ""

# ========================
# Derived dataset config
# ========================
if DATASET_SHORT == "trec-covid":
    INDEX_NAME  = "beir-v1.0.0-trec-covid.flat"
    TOPICS_NAME = "beir-v1.0.0-trec-covid-test"
elif DATASET_SHORT == "nfcorpus":
    INDEX_NAME  = "beir-v1.0.0-nfcorpus.flat"
    TOPICS_NAME = "beir-v1.0.0-nfcorpus-test"
else:
    raise ValueError("DATASET_SHORT must be 'trec-covid' or 'nfcorpus'")

SOURCE_ROOT = f"results/{DATASET_SHORT}_{SOURCE_MODEL_SHORT}"
TARGET_ROOT = f"results/{DATASET_SHORT}_{TARGET_MODEL_SHORT}"

# ========================
# Helpers
# ========================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def iter_input_files(mode_dir: str) -> Dict[str, Dict[int, str]]:
    """
    Walk a mode_dir (e.g., results/ds_model/N10/half_relevant) and return:
      { qid: { input_index : path_to_run0_json } }
    We only key off run0 to get the canonical inputs; run1 will be generated later.
    """
    mapping: Dict[str, Dict[int, str]] = {}
    if not os.path.isdir(mode_dir):
        return mapping
    for qid in os.listdir(mode_dir):
        qpath = os.path.join(mode_dir, qid)
        if not os.path.isdir(qpath):
            continue
        for fname in os.listdir(qpath):
            # expect inXX_run0.json
            m = re.match(r"in(\d{2})_run0\.json$", fname)
            if not m:
                continue
            inp_idx = int(m.group(1))
            mapping.setdefault(qid, {})[inp_idx] = os.path.join(qpath, fname)
    return mapping

def robust_qrels(qrels_raw: dict, qid: str) -> Dict[str, int]:
    """Return qrel dict for qid, tolerant of str/int key types."""
    if qid in qrels_raw:
        return qrels_raw[qid]
    try:
        qi = int(qid)
        return qrels_raw.get(qi, {})
    except:
        return qrels_raw.get(str(qid), {})

def count_rel_nonrel(docids: List[str], qrel_dict: Dict[str, int]) -> Tuple[int,int]:
    rel = sum(1 for d in docids if int(qrel_dict.get(d, 0)) > 0)
    return rel, len(docids) - rel

def verify_mode_counts(mode: str, size: int, docids: List[str], qrel_dict: Dict[str,int]) -> Tuple[bool, str]:
    """Check expected composition per mode. Return (ok, msg)."""
    rel, non = count_rel_nonrel(docids, qrel_dict)
    if mode == "single_relevant":
        ok = (rel == 1 and non == size - 1)
        return ok, f"[single_relevant] N{size}: rel={rel} nonrel={non} (expected 1 / {size-1})"
    if mode == "single_nonrelevant":
        ok = (rel == size - 1 and non == 1)
        return ok, f"[single_nonrelevant] N{size}: rel={rel} nonrel={non} (expected {size-1} / 1)"
    if mode == "half_relevant":
        # allow ±1 slack (as in your previous design)
        want = size // 2
        ok = (want - 1) <= rel <= (want + 1)
        return ok, f"[half_relevant] N{size}: rel={rel} nonrel={non} (target ≈ {want}/{size-want})"
    return True, f"[unknown mode] N{size}: rel={rel} nonrel={non}"

def verify_subset_chain(doc5: Optional[List[str]],
                        doc10: Optional[List[str]],
                        doc20: Optional[List[str]]) -> Tuple[bool, str]:
    """Check N5 ⊆ N10 ⊆ N20 when applicable."""
    if doc5 and doc10:
        if not set(doc5).issubset(set(doc10)):
            return False, "N5 ⊄ N10"
    if doc10 and doc20:
        if not set(doc10).issubset(set(doc20)):
            return False, "N10 ⊄ N20"
    if doc5 and doc20:
        if not set(doc5).issubset(set(doc20)):
            return False, "N5 ⊄ N20"
    return True, "subset chain OK"

def build_hits_from_docids(searcher: LuceneSearcher, docids: List[str]) -> List[Dict[str,str]]:
    """Fetch raw contents for docids in-order. Skips any missing doc (warn)."""
    hits = []
    missing = []
    for d in docids:
        dobj = searcher.doc(d)
        if dobj is None:
            missing.append(d)
            continue
        hits.append({"docid": d, "content": dobj.raw()})
    if missing:
        print(f"  ⚠️  Missing {len(missing)} docs in index; will skip them: {missing[:3]}{'...' if len(missing)>3 else ''}")
    return hits

def target_log_exists(target_qdir: str, inp_idx: int, run_idx: int) -> bool:
    return os.path.exists(os.path.join(target_qdir, f"in{inp_idx:02d}_run{run_idx}.json"))

# ========================
# MAIN
# ========================

def main():
    print(f"=== Reusing GPT-4 inputs → run with {TARGET_MODEL_NAME} on {DATASET_SHORT}")
    print(f"Source: {SOURCE_ROOT}")
    print(f"Target: {TARGET_ROOT}")
    ensure_dir(TARGET_ROOT)

    topics = get_topics(TOPICS_NAME)
    qrels  = get_qrels(TOPICS_NAME)
    searcher = LuceneSearcher.from_prebuilt_index(INDEX_NAME)

    # Gather source inputs across sizes/modes
    # src_map[size][mode] -> { qid: {inp_idx: path_to_run0} }
    src_map: Dict[int, Dict[str, Dict[str, Dict[int, str]]]] = {}
    for size in SIZES:
        size_dir = os.path.join(SOURCE_ROOT, f"N{size}")
        size_modes = {}
        for mode in MODES:
            mode_dir = os.path.join(size_dir, mode)
            size_modes[mode] = iter_input_files(mode_dir)
        src_map[size] = size_modes

    # Derive qid set per mode: union of all sizes present under SOURCE
    for mode in MODES:
        # Union any qids that appear in any size for this mode
        qids = set()
        for size in SIZES:
            qids.update(src_map[size][mode].keys())
        if not qids:
            print(f"[{mode}] No source inputs found. Skipping.")
            continue

        print(f"\n=== Mode: {mode} | qids={len(qids)} ===")
        for qid in tqdm(sorted(qids, key=lambda x: (len(x), x))):
            qrel_dict = robust_qrels(qrels, qid)

            # Build a per-input view across sizes using the indices present in SOURCE
            # We will consider the union of input indices seen in any size for this qid
            inp_idcs = set()
            for size in SIZES:
                inp_idcs.update(src_map[size][mode].get(qid, {}).keys())
            if not inp_idcs:
                continue

            for inp_idx in sorted(inp_idcs):
                # Load source input_docids per size (if exists)
                src_inputs: Dict[int, List[str]] = {}
                for size in SIZES:
                    p = src_map[size][mode].get(qid, {}).get(inp_idx)
                    if not p:
                        continue
                    js = load_json(p)
                    if not js or "input_docids" not in js:
                        continue
                    src_inputs[size] = list(js["input_docids"])

                if not src_inputs:
                    continue

                # ---- verifications before LLM ----
                # counts per size
                ok_counts_all = True
                for size, docids in src_inputs.items():
                    ok_counts, msg_counts = verify_mode_counts(mode, size, docids, qrel_dict)
                    if not ok_counts:
                        ok_counts_all = False
                        print(f"  [COUNT MISMATCH] Q{qid} in={inp_idx:02d} {msg_counts}")
                # subset chain
                doc5  = src_inputs.get(5)
                doc10 = src_inputs.get(10)
                doc20 = src_inputs.get(20)
                ok_subset, msg_subset = verify_subset_chain(doc5, doc10, doc20)
                if not ok_subset:
                    print(f"  [SUBSET WARN]   Q{qid} in={inp_idx:02d}: {msg_subset}")
                else:
                    # Only print pass succinctly
                    pass

                if CHECK_ONLY:
                    # Just verifying inputs; do not call LLM
                    continue

                # ---- run LLM per size (re-using EXACT source docids order) ----
                for size, docids in sorted(src_inputs.items()):
                    # Target output folder mirrors source
                    target_qdir = os.path.join(TARGET_ROOT, f"N{size}", mode, str(qid))
                    ensure_dir(target_qdir)

                    # If both run0 and run1 already exist, skip
                    if all(target_log_exists(target_qdir, inp_idx, r) for r in range(DETER_RUNS)):
                        continue

                    # Rehydrate hits from docids
                    hits = build_hits_from_docids(searcher, docids)
                    if len(hits) < 2:
                        # Not enough for meaningful permutation; skip politely
                        print(f"  ⚠️  Q{qid} in={inp_idx:02d} N{size}: only {len(hits)} valid hits (skip)")
                        continue

                    # Fetch query text (topics may have str or int keys)
                    qkey = qid if qid in topics else (int(qid) if str(qid).isdigit() and int(qid) in topics else str(qid))
                    query_text = topics[qkey]["title"]

                    # Two deterministic runs
                    for run_idx in range(DETER_RUNS):
                        if target_log_exists(target_qdir, inp_idx, run_idx):
                            continue
                        item = {"query": query_text, "hits": copy.deepcopy(hits)}
                        msgs = create_permutation_instruction(item, 0, len(hits), model_name=TARGET_MODEL_NAME)
                        resp = run_llm(msgs, api_key=API_KEY, model_name=TARGET_MODEL_NAME)
                        out  = receive_permutation(item, resp, 0, len(hits))

                        log = {
                            "qid": qid,
                            "mode": mode,
                            "size": size,
                            "input_index": inp_idx,
                            "run": run_idx,
                            "prompt": msgs,
                            "input_docids": [h["docid"] for h in hits],  # SAME ORDER as source inputs
                            "output_docids": [h["docid"] for h in out["hits"][:len(hits)]],
                            "response": resp
                        }
                        with open(os.path.join(target_qdir, f"in{inp_idx:02d}_run{run_idx}.json"), "w") as f:
                            json.dump(log, f, indent=2)

    print("\n✅ Done. Target logs under:", TARGET_ROOT)
    if CHECK_ONLY:
        print("ℹ️  CHECK_ONLY=True: LLM was not called; only input parity verified.")

if __name__ == "__main__":
    main()
