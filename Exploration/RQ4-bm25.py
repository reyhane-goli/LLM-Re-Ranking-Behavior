import os
import json
import copy
from math import log2
from collections import defaultdict
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels
from rank_gpt import create_permutation_instruction, run_llm, receive_permutation

# ─── CONFIG ────────────────────────────────────────────────────────────────
INDEX_NAME     = 'beir-v1.0.0-trec-covid.flat'
TOPICS_NAME    = 'beir-v1.0.0-trec-covid-test'
OUTPUT_DIR     = "scenario6_logs_Second_Time"
API_KEY        = ""
TOP_K          = 10
RUNS_PER_INPUT = 1  # Only run each model once per input

# LLM model  - only run once in a time gpt 3 OR gpt 4 
MODELS = {
#     "gpt-3.5-turbo": API_KEY,  
    "gpt-4": API_KEY,         
}

# Input ordering modes
INPUT_ORDERS = {
    "score_desc": lambda docs: sorted(docs, key=lambda d: -d['score']),
    "docid_asc":  lambda docs: sorted(docs, key=lambda d: d['docid']),
    "score_asc":  lambda docs: sorted(docs, key=lambda d: d['score']),
}

# ─── FUNCTIONS ─────────────────────────────────────────────────────────────
def dcg_at_k(rels, k=TOP_K):
    return sum(rel / log2(i + 2) for i, rel in enumerate(rels[:k]))

def ndcg_for_out(out_docs, qrel_dict, k=TOP_K):
    labels = [int(qrel_dict.get(d, 0)) for d in out_docs[:k]]
    dcg = dcg_at_k(labels, k)
    ideal = sorted(labels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0

# ─── Load Data ─────────────────────────────────────────────────────────────
topics = get_topics(TOPICS_NAME)
qrels = get_qrels(TOPICS_NAME)
searcher = LuceneSearcher.from_prebuilt_index(INDEX_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ndcg_scores = defaultdict(lambda: defaultdict(list))  # model → input_order → list of scores
bm25_scores_by_order = defaultdict(list)

# ─── Main Loop ─────────────────────────────────────────────────────────────
for qid in tqdm(sorted(topics.keys(), key=int)):  # ← Only first 2 queries
    query = topics[qid]['title']
    hits = searcher.search(query, k=TOP_K)
    raw_docs = [{"docid": h.docid, "score": h.score, "content": searcher.doc(h.docid).raw()} for h in hits]

    for order_name, reorder_func in INPUT_ORDERS.items():

        ordered_docs = reorder_func(copy.deepcopy(raw_docs))
        ordered_docids = [d["docid"] for d in ordered_docs]

        # Compute and store BM25 NDCG
        score = ndcg_for_out(ordered_docids, qrels[qid])
        bm25_scores_by_order[order_name].append(score)

        # Save input docids
        qdir = os.path.join(OUTPUT_DIR, order_name, str(qid))
        os.makedirs(qdir, exist_ok=True)
        with open(os.path.join(qdir, "bm25_input.json"), "w") as f:
            json.dump(ordered_docids, f, indent=2)

        for model_name, api_key in MODELS.items():
            item = {"query": query, "hits": ordered_docs}
            msgs = create_permutation_instruction(item, 0, len(ordered_docs), model_name=model_name)
            resp = run_llm(msgs, api_key=api_key, model_name=model_name)
            out = receive_permutation(item, resp, 0, len(ordered_docs))
            output_docids = [d["docid"] for d in out["hits"][:len(ordered_docs)]]

            # Save log
            log = {
                "qid": qid,
                "input_order": order_name,
                "model": model_name,
                "run": 0,
                "prompt": msgs,
                "input_docids": ordered_docids,
                "output_docids": output_docids,
                "response": resp
            }
            with open(os.path.join(qdir, f"{model_name}_run0.json"), "w") as f:
                json.dump(log, f, indent=2)

            # Score LLM reranking
            ndcg = ndcg_for_out(output_docids, qrels[qid])
            ndcg_scores[model_name][order_name].append(ndcg)

# # ─── Save NDCG Summary Table ───────────────────────────────────────────────
# summary_path = os.path.join(OUTPUT_DIR, "ndcg_summary.txt")
# with open(summary_path, "w") as f:
#     f.write("Input Order         BM25       GPT-3.5-turbo     GPT-4\n")
#     f.write("=" * 60 + "\n")
#     for order_name in INPUT_ORDERS:
#         bm25_mean = sum(bm25_scores_by_order[order_name]) / len(bm25_scores_by_order[order_name])
#         line = f"{order_name:20}{bm25_mean:10.4f}"
#         for model_name in MODELS:
#             scores = ndcg_scores[model_name][order_name]
#             mean_score = sum(scores) / len(scores) if scores else 0.0
#             line += f"{mean_score:18.4f}"
#         f.write(line + "\n")



# import os, json
# from math import log2
# from collections import defaultdict
# import pandas as pd
# from scipy.stats import wilcoxon
# from pyserini.search import get_qrels

# # ─── CONFIG ────────────────────────────────────────────────────────────────
# SCEN6_DIR   = "scenario6_logs"                 # root of your logs
# TOPICS     = 'beir-v1.0.0-trec-covid-test'
# OUTPUT     = "significance_results.txt"
# TOP_K      = 10

# # ─── METRICS ───────────────────────────────────────────────────────────────
# def dcg_at_k(rels, k=TOP_K):
#     return sum(rel/log2(i+2) for i,rel in enumerate(rels[:k]))

# def ndcg_for_out(ids, qrel):
#     labs  = [int(qrel.get(d,0)) for d in ids[:TOP_K]]
#     idcg  = dcg_at_k(sorted(labs, reverse=True), TOP_K)
#     return dcg_at_k(labs, TOP_K)/idcg if idcg>0 else 0.0

# def ndcg_full_qrel(ids, qrel, k=TOP_K):
#     # 1) DCG of your output
#     labs = [int(qrel.get(d, 0)) for d in ids[:k]]
#     dcg = dcg_at_k(labs, k)

#     # 2) IDCG over all qrel relevances
#     all_rels = sorted(qrel.values(), reverse=True)
#     idcg = dcg_at_k(all_rels, k)

#     return dcg / idcg if idcg > 0 else 0.0


# # ─── LOAD qrels ────────────────────────────────────────────────────────────
# raw_qrels = get_qrels(TOPICS)
# # qrels = {str(k):v for k,v in qrels.items()}  # ensure string keys
# qrels = {
#     str(qid): { docid: int(rel) for docid, rel in relmap.items() }
#     for qid, relmap in raw_qrels.items()
# }

# # ─── LOAD SCORES ───────────────────────────────────────────────────────────
# rows = []
# for order in os.listdir(SCEN6_DIR):
#     p_order = os.path.join(SCEN6_DIR, order)
#     if not os.path.isdir(p_order): 
#         continue
#     for qid in sorted(os.listdir(p_order), key=int):
#         p_q = os.path.join(p_order, qid)

#         # BM25 baseline
#         f0 = os.path.join(p_q, "bm25_input.json")
#         if os.path.exists(f0):
#             ids = json.load(open(f0))
# #             rows.append({"order":order, "qid":int(qid),
# #                          "model":"none", "ndcg":ndcg_for_out(ids, qrels[qid])})
#             rows.append({"order":order, "qid":int(qid),
#                          "model":"none", "ndcg":ndcg_full_qrel(ids, qrels[qid])})

#         # GPT outputs
#         for mdl in ("gpt-3.5-turbo","gpt-4"):
#             fm = os.path.join(p_q, f"{mdl}_run0.json")
#             if not os.path.exists(fm): continue
#             out = json.load(open(fm))["output_docids"]
# #             rows.append({"order":order, "qid":int(qid),
# #                          "model":mdl, "ndcg":ndcg_for_out(out, qrels[qid])})
#             rows.append({"order":order, "qid":int(qid),
#                          "model":mdl, "ndcg":ndcg_full_qrel(out, qrels[qid])})

# df = pd.DataFrame(rows)



########################## test #########################
# Focus only on the BM25‐desc input order
# sub = df[df["order"] == "score_desc"]

# Compute mean nDCG@10 for each model under that order
# mean_by_model = df.groupby("model")["ndcg"].mean()

# print("Mean nDCG@10 for score_desc (actual BM25 order):")
# for model, val in mean_by_model.items():
#     print(f"  {model}: {val:.4f}")


# mean_ndcg = (
#     df
#     .groupby(["order", "model"])["ndcg"]
#     .mean()
#     .unstack()  # makes a table with orders as rows, models as columns
# )

# print("Mean NDCG@10 by input order and reranker:")
# print(mean_ndcg)


# # ─── PAIRS TO TEST ─────────────────────────────────────────────────────────
# reranker_pairs = [
#     ("gpt-3.5-turbo","none"),
#     ("gpt-4","none"),
#     ("gpt-3.5-turbo","gpt-4"),
#     ("gpt-4","gpt-3.5-turbo"),
# ]
# order_pairs = [
#     ("score_desc","docid_asc"),
#     ("score_desc","score_asc"),
#     ("docid_asc","score_asc"),
#     ("docid_asc","score_desc"),
# ]

# # ─── RUN TESTS ─────────────────────────────────────────────────────────────
# with open(OUTPUT, "w") as out:
#     out.write("One‑sided Wilcoxon Tests (A > B)\n")
#     out.write("="*40 + "\n\n")

#     out.write("▶ RERANKER COMPARISONS (within each input order)\n")
#     for order in df["order"].unique():
#         sub = df[df["order"]==order]
#         out.write(f"\n[{order}]\n")
#         for A,B in reranker_pairs:
#             a = sub[sub["model"]==A].sort_values("qid")["ndcg"]
#             b = sub[sub["model"]==B].sort_values("qid")["ndcg"]
#             if len(a)==len(b)>0:
#                 p = wilcoxon(a,b,alternative="greater").pvalue
#                 sig = "✔" if p<0.05 else "—"
#                 out.write(f"  {A:15} > {B:15} : p={p:.4f} {sig}\n")

#     out.write("\n▶ INPUT‑ORDER COMPARISONS (within each model)\n")
#     for model in df["model"].unique():
#         sub = df[df["model"]==model]
#         out.write(f"\n[{model}]\n")
#         for O1,O2 in order_pairs:
#             a = sub[sub["order"]==O1].sort_values("qid")["ndcg"]
#             b = sub[sub["order"]==O2].sort_values("qid")["ndcg"]
#             if len(a)==len(b)>0:
#                 p = wilcoxon(a,b,alternative="greater").pvalue
#                 sig = "✔" if p<0.05 else "—"
#                 out.write(f"  {O1:12} > {O2:12} : p={p:.4f} {sig}\n")

# print(f"✅ Written to {OUTPUT}")






import os, json
from math import log2
import pandas as pd
from scipy.stats import wilcoxon
from pyserini.search import get_qrels

# ─── CONFIG ────────────────────────────────────────────────────────────────
SCEN6_DIR   = "scenario6_logs_Second_Time"                 # root of your logs
TOPICS     = 'beir-v1.0.0-trec-covid-test'
SUM_PATH   = os.path.join(SCEN6_DIR, "ndcg_summary.txt")
TEST_PATH  = os.path.join(SCEN6_DIR, "significance_tests.txt")
TOP_K      = 10

# ─── METRICS ───────────────────────────────────────────────────────────────
def dcg_at_k(rels, k=TOP_K):
    return sum(rel/log2(i+2) for i,rel in enumerate(rels[:k]))

def ndcg_full_qrel(ids, qrel, k=TOP_K):
    labs    = [int(qrel.get(d,0)) for d in ids[:k]]
    dcg_val = dcg_at_k(labs, k)
    all_rels = sorted(qrel.values(), reverse=True)
    idcg_val = dcg_at_k(all_rels, k)
    return dcg_val/idcg_val if idcg_val>0 else 0.0

# ─── LOAD QRELS ────────────────────────────────────────────────────────────
raw_qrels = get_qrels(TOPICS)
qrels = { str(qid): {doc:int(rel) for doc,rel in relmap.items()} 
          for qid,relmap in raw_qrels.items() }

# ─── COLLECT SCORES ─────────────────────────────────────────────────────────
rows = []
for order in ("score_desc","docid_asc","score_asc"):
    p_order = os.path.join(SCEN6_DIR, order)
    if not os.path.isdir(p_order): continue

    for qid in sorted(os.listdir(p_order), key=int):
        p_q = os.path.join(p_order, qid)

        # BM25 baseline
        f0 = os.path.join(p_q, "bm25_input.json")
        if os.path.exists(f0):
            ids = json.load(open(f0))
            rows.append({
                "order":order, "qid":int(qid),
                "model":"none",
                "ndcg": ndcg_full_qrel(ids, qrels[qid])
            })

        # GPT outputs
        for mdl in ("gpt-3.5-turbo","gpt-4"):
            fm = os.path.join(p_q, f"{mdl}_run0.json")
            if not os.path.exists(fm): continue
            out_ids = json.load(open(fm))["output_docids"]
            rows.append({
                "order":order, "qid":int(qid),
                "model":mdl,
                "ndcg": ndcg_full_qrel(out_ids, qrels[qid])
            })

df = pd.DataFrame(rows)

# ─── SAVE NDCG SUMMARY ─────────────────────────────────────────────────────
models = ["none","gpt-3.5-turbo","gpt-4"]
with open(SUM_PATH, "w") as f:
    f.write("Input Order         BM25       GPT-3.5-turbo     GPT-4\n")
    f.write("="*60 + "\n")
    for order in ("score_desc","docid_asc","score_asc"):
        grp = df[df["order"]==order].groupby("model")["ndcg"].mean()
        bm25_m = grp.get("none", 0.0)
        t35_m  = grp.get("gpt-3.5-turbo", 0.0)
        g4_m   = grp.get("gpt-4", 0.0)
        f.write(f"{order:20}{bm25_m:10.4f}{t35_m:18.4f}{g4_m:18.4f}\n")

# ─── RUN WILCOXON TESTS ────────────────────────────────────────────────────
reranker_pairs = [
    ("gpt-3.5-turbo","none"),
    ("gpt-4","none"),
    ("gpt-3.5-turbo","gpt-4"),
    ("gpt-4","gpt-3.5-turbo"),
]
order_pairs = [
    ("score_desc","docid_asc"),
    ("score_desc","score_asc"),
    ("docid_asc","score_asc"),
    ("docid_asc","score_desc"),
]

with open(TEST_PATH, "w") as f:
    f.write("One‑sided Wilcoxon Tests (A > B)\n")
    f.write("="*40 + "\n\n")

    f.write("▶ RERANKER COMPARISONS (within each input order)\n")
    for order in ("score_desc","docid_asc","score_asc"):
        sub = df[df["order"]==order]
        f.write(f"\n[{order}]\n")
        for A,B in reranker_pairs:
            a = sub[sub["model"]==A].sort_values("qid")["ndcg"]
            b = sub[sub["model"]==B].sort_values("qid")["ndcg"]
            if len(a)==len(b)>0:
                p = wilcoxon(a,b,alternative="greater").pvalue
                sig = "✔" if p<0.05 else "—"
                f.write(f"  {A:15} > {B:15} : p={p:.4f} {sig}\n")

    f.write("\n▶ INPUT‑ORDER COMPARISONS (within each model)\n")
    for mdl in ("none","gpt-3.5-turbo","gpt-4"):
        sub = df[df["model"]==mdl]
        f.write(f"\n[{mdl}]\n")
        for O1,O2 in order_pairs:
            a = sub[sub["order"]==O1].sort_values("qid")["ndcg"]
            b = sub[sub["order"]==O2].sort_values("qid")["ndcg"]
            if len(a)==len(b)>0:
                p = wilcoxon(a,b,alternative="greater").pvalue
                sig = "✔" if p<0.05 else "—"
                f.write(f"  {O1:12} > {O2:12} : p={p:.4f} {sig}\n")

print(f"✅ NDCG summary in `{SUM_PATH}`")
print(f"✅ Significance tests in `{TEST_PATH}`")
