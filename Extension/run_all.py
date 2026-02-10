import os, argparse, pandas as pd
from config import *
from utils import TeeLogger, ts_filename, timestamp
from retrieve_bm25 import retrieve_topk
from rerank_supervised import MonoBERTReRanker, MonoT5ReRanker, rerank_with_progress as rerank_supervised
from rerank_unsupervised import CrossEncoderReRanker, T5YesNoScorer, rerank_with_progress as rerank_unsup
# from rerank_llm import rerank_with_progress as rerank_llm
from evaluate import results_to_run, eval_ndcg10
from tqdm import tqdm

def get_doc_fetcher(searcher):
    from rerank_supervised import _get_text_from_raw
    def fetch(docid: str) -> str:
        d = searcher.doc(docid)
        if d is None: return ""
        return _get_text_from_raw(d.raw())
    return fetch

# def run():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--datasets", nargs="+", required=True)
#     parser.add_argument("--methods", nargs="+", required=True)
#     parser.add_argument("--max-docs", type=int, default=100)
#     parser.add_argument("--batch-size", type=int, default=8)
#     parser.add_argument("--output", type=str, default="results.csv")
#     args = parser.parse_args()

#     os.makedirs(RESULTS_DIR, exist_ok=True)
#     os.makedirs(LOGS_DIR, exist_ok=True)

#     log_path = os.path.join(LOGS_DIR, f"run_{ts_filename()}.txt")
#     tee = TeeLogger(log_path)
#     print(f"[INFO {timestamp()}] Logging to {log_path}", flush=True)

#     rows = []

#     for dataset in args.datasets:
#         print(f"\n=== {dataset} :: BM25 top-{args.max_docs} ===", flush=True)
#         bm25_res, qrels, queries, searcher = retrieve_topk(dataset, k=args.max_docs)
#         nd_bm25 = eval_ndcg10(results_to_run(bm25_res), qrels)
#         rows.append((dataset, "BM25", nd_bm25))
#         print(f"BM25 nDCG@10 = {nd_bm25:.4f}", flush=True)
#         fetch = get_doc_fetcher(searcher)

#         # Supervised
#         if "monoBERT" in args.methods:
#             model = MonoBERTReRanker(MONOBERT_CKPT)
#             rr = rerank_supervised("monoBERT", model, bm25_res, queries, fetch, batch_size=args.batch_size, colour="blue")
#             nd = eval_ndcg10(results_to_run(rr), qrels)
#             rows.append((dataset, "monoBERT (340M)", nd)); print(f"monoBERT nDCG@10 = {nd:.4f}", flush=True)

#         if "monoT5_BASE" in args.methods:
#             model = MonoT5ReRanker(MONOT5_BASE_CKPT)
#             rr = rerank_supervised("monoT5-base", model, bm25_res, queries, fetch, batch_size=args.batch_size, colour="blue")
#             nd = eval_ndcg10(results_to_run(rr), qrels)
#             rows.append((dataset, "monoT5 (220M)", nd)); print(f"monoT5-base nDCG@10 = {nd:.4f}", flush=True)

#         if "monoT5_3B" in args.methods:
#             model = MonoT5ReRanker(MONOT5_3B_CKPT)
#             rr = rerank_supervised("monoT5-3B", model, bm25_res, queries, fetch, batch_size=max(1,args.batch_size//2), colour="blue")
#             nd = eval_ndcg10(results_to_run(rr), qrels)
#             rows.append((dataset, "monoT5 (3B)", nd)); print(f"monoT5-3B nDCG@10 = {nd:.4f}", flush=True)

#         # Un/supervised CE & T5-based
#         if "mmarcoCE" in args.methods:
#             model = CrossEncoderReRanker(MMARCO_CE_CKPT)
#             rr = rerank_unsup("mmarcoCE", model, bm25_res, queries, fetch, batch_size=args.batch_size*2, colour="yellow")
#             nd = eval_ndcg10(results_to_run(rr), qrels)
#             rows.append((dataset, "mmarcoCE", nd)); print(f"mmarcoCE nDCG@10 = {nd:.4f}", flush=True)

# #         if "upr" in args.methods:
# #             model = T5YesNoScorer(UPR_FLAN_CKPT)
# #             rr = rerank_unsup("UPR", model, bm25_res, queries, fetch, batch_size=max(1,args.batch_size//2), colour="yellow")
# #             nd = eval_ndcg10(results_to_run(rr), qrels)
# #             rows.append((dataset, "UPR (FLAN-T5)", nd)); print(f"UPR nDCG@10 = {nd:.4f}", flush=True)

#         if "upr" in args.methods:

#             from rerank_upr_fixed import UPRReRanker
#             print("[INFO] Running UPR (FLAN-T5-XL, fixed prompt)...")
#             upr = UPRReRanker("google/flan-t5-xl")
#             ndcg_values = []
#             for qid, qtext in tqdm(queries.items()):
#                 docs = bm25_res[qid][:args.max_docs]
#                 reranked = upr.rerank(qtext, docs, searcher)
#                 # --- convert to BEIR format ---
#                 run_dict = {qid: {docid: score for docid, score in reranked}}
#                 # Safely handle missing qids
#                 if qid not in qrels:
#                     print(f"[WARN] Skipping qid {qid}: no relevance judgments found.")
#                     continue
#                 qrels_dict = {qid: qrels[qid]}
#                 ndcg_values.append(eval_ndcg10(run_dict, qrels_dict))
#             print(f"UPR (FLAN-T5-XL) nDCG@10 = {sum(ndcg_values)/len(ndcg_values):.4f}")

    

#         if "inpars" in args.methods:
#             model = T5YesNoScorer(INPARS_CKPT)
#             rr = rerank_unsup("InPars", model, bm25_res, queries, fetch, batch_size=max(1,args.batch_size//2), colour="yellow")
#             nd = eval_ndcg10(results_to_run(rr), qrels)
#             rows.append((dataset, "InPars (monoT5)", nd)); print(f"InPars nDCG@10 = {nd:.4f}", flush=True)

#         if "promptagator" in args.methods:
#             model = T5YesNoScorer(PROMPTAGATOR_CKPT)
#             rr = rerank_unsup("Promptagator++", model, bm25_res, queries, fetch, batch_size=args.batch_size, colour="yellow")
#             nd = eval_ndcg10(results_to_run(rr), qrels)
#             rows.append((dataset, "Promptagator++", nd)); print(f"Promptagator++ nDCG@10 = {nd:.4f}", flush=True)

#         # API LLMs
#         if "gpt35" in args.methods:
#             rr = rerank_llm("gpt35", bm25_res, queries, fetch, batch_size=args.batch_size, openai_model=OPENAI_MODEL_GPT35, colour="green")
#             nd = eval_ndcg10(results_to_run(rr), qrels)
#             rows.append((dataset, "gpt-3.5-turbo", nd)); print(f"gpt-3.5 nDCG@10 = {nd:.4f}", flush=True)

#         if "gpt4" in args.methods:
#             rr = rerank_llm("gpt4", bm25_res, queries, fetch, batch_size=max(1,args.batch_size//2), openai_model=OPENAI_MODEL_GPT4, colour="green")
#             nd = eval_ndcg10(results_to_run(rr), qrels)
#             rows.append((dataset, "gpt-4", nd)); print(f"gpt-4 nDCG@10 = {nd:.4f}", flush=True)

#         if "cohere" in args.methods:
#             rr = rerank_llm("cohere", bm25_res, queries, fetch, batch_size=args.batch_size, colour="green")
#             nd = eval_ndcg10(results_to_run(rr), qrels)
#             rows.append((dataset, "Cohere Rerank", nd)); print(f"Cohere nDCG@10 = {nd:.4f}", flush=True)

#     df = pd.DataFrame(rows, columns=["Dataset","Method","nDCG@10"])
#     pivot = df.pivot_table(index="Method", columns="Dataset", values="nDCG@10", aggfunc="mean")
#     out_csv = os.path.join(RESULTS_DIR, os.path.basename(args.output))
#     pivot.to_csv(out_csv)
#     print(f"\nSaved table to {out_csv}", flush=True)
#     print(f"[INFO {timestamp()}] Logs saved to: {os.path.abspath(log_path)}", flush=True)
#     with pd.option_context('display.max_columns', None, 'display.width', 160):
#         print("\n=== Summary (nDCG@10) ===")
#         print(pivot.fillna('-'))
        
        
#     if len(args.datasets) > 1:
#         avg_ndcg = np.mean(ndcg_values)
#         print(f"\n=== BEIR (Avg) over {len(args.datasets)} datasets ===")
#         print(f"BEIR Avg nDCG@10 = {avg_ndcg:.4f}")

#     tee.close()


def run():
    import traceback
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--max-docs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=str, default="results.csv")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    log_path = os.path.join(LOGS_DIR, f"run_{ts_filename()}.txt")
    tee = TeeLogger(log_path)
    print(f"[INFO {timestamp()}] Logging to {log_path}", flush=True)

    # also maintain a per-run text result file
    txt_log_path = os.path.join(LOGS_DIR, f"results_{ts_filename()}.txt")

    rows = []

    for dataset in args.datasets:
        print(f"\n=== {dataset} :: BM25 top-{args.max_docs} ===", flush=True)
        try:
            bm25_res, qrels, queries, searcher = retrieve_topk(dataset, k=args.max_docs)
            nd_bm25 = eval_ndcg10(results_to_run(bm25_res), qrels)
            rows.append((dataset, "BM25", nd_bm25))
            print(f"BM25 nDCG@10 = {nd_bm25:.4f}", flush=True)
            fetch = get_doc_fetcher(searcher)
        except Exception as e:
            print(f"[ERROR] Failed during BM25 retrieval for {dataset}: {e}")
            traceback.print_exc()
            continue

        # Iterate over all requested methods robustly
        for method in args.methods:
            print(f"\n[INFO] Starting {method} on {dataset} ...", flush=True)
            try:
                nd = None  # will store final nDCG
                if method == "monoBERT":
#                     model = MonoBERTReRanker(MONOBERT_CKPT)
                    model = MonoBERTReRanker(MONOBERT_CKPT, max_len=512)
                    rr = rerank_supervised("monoBERT", model, bm25_res, queries, fetch,
                                           batch_size=args.batch_size, colour="blue")
                    nd = eval_ndcg10(results_to_run(rr), qrels)

                elif method == "monoT5_BASE":
                    model = MonoT5ReRanker(MONOT5_BASE_CKPT)
                    rr = rerank_supervised("monoT5-base", model, bm25_res, queries, fetch,
                                           batch_size=args.batch_size, colour="blue")
                    nd = eval_ndcg10(results_to_run(rr), qrels)

                elif method == "monoT5_3B":
                    model = MonoT5ReRanker(MONOT5_3B_CKPT)
                    rr = rerank_supervised("monoT5-3B", model, bm25_res, queries, fetch,
                                           batch_size=max(1, args.batch_size // 2), colour="blue")
                    nd = eval_ndcg10(results_to_run(rr), qrels)

                elif method == "mmarcoCE":
                    model = CrossEncoderReRanker(MMARCO_CE_CKPT)
                    rr = rerank_unsup("mmarcoCE", model, bm25_res, queries, fetch,
                                      batch_size=args.batch_size * 2, colour="yellow")
                    nd = eval_ndcg10(results_to_run(rr), qrels)

                elif method == "upr":
                    from rerank_upr_fixed import UPRReRanker
                    print("[INFO] Running UPR (FLAN-T5-XL, fixed prompt)...", flush=True)
#                     upr = UPRReRanker("google/flan-t5-xl")
                    upr = UPRReRanker("castorini/monot5-large-msmarco-10k")
                    ndcg_values = []
                    for qid, qtext in tqdm(queries.items(), desc=f"{dataset}-UPR"):
                        docs = bm25_res[qid][:args.max_docs]
                        reranked = upr.rerank(qtext, docs, searcher)
                        if qid not in qrels:
                            print(f"[WARN] Skipping qid {qid}: no relevance judgments found.")
                            continue
                        run_dict = {qid: {docid: score for docid, score in reranked}}
                        qrels_dict = {qid: qrels[qid]}
                        ndcg_values.append(eval_ndcg10(run_dict, qrels_dict))
                    if len(ndcg_values) > 0:
                        nd = sum(ndcg_values) / len(ndcg_values)
                    else:
                        nd = 0.0

                elif method == "inpars":
                    model = T5YesNoScorer(INPARS_CKPT)
                    rr = rerank_unsup("InPars", model, bm25_res, queries, fetch,
                                      batch_size=max(1, args.batch_size // 2), colour="yellow")
                    nd = eval_ndcg10(results_to_run(rr), qrels)

                elif method == "promptagator":
                    model = T5YesNoScorer(PROMPTAGATOR_CKPT)
                    rr = rerank_unsup("Promptagator++", model, bm25_res, queries, fetch,
                                      batch_size=args.batch_size, colour="yellow")
                    nd = eval_ndcg10(results_to_run(rr), qrels)

                elif method == "gpt35":
                    rr = rerank_llm("gpt35", bm25_res, queries, fetch,
                                    batch_size=args.batch_size, openai_model=OPENAI_MODEL_GPT35, colour="green")
                    nd = eval_ndcg10(results_to_run(rr), qrels)

                elif method == "gpt4":
                    rr = rerank_llm("gpt4", bm25_res, queries, fetch,
                                    batch_size=max(1, args.batch_size // 2), openai_model=OPENAI_MODEL_GPT4, colour="green")
                    nd = eval_ndcg10(results_to_run(rr), qrels)

                elif method == "cohere":
                    rr = rerank_llm("cohere", bm25_res, queries, fetch,
                                    batch_size=args.batch_size, colour="green")
                    nd = eval_ndcg10(results_to_run(rr), qrels)

                else:
                    print(f"[WARN] Unknown method '{method}', skipping.")
                    continue

                # --- log success ---
                if nd is not None:
                    rows.append((dataset, method, nd))
                    print(f"[RESULT] {dataset} | {method} | nDCG@10 = {nd:.4f}", flush=True)
                    with open(txt_log_path, "a") as f:
                        f.write(f"{timestamp()} | {dataset} | {method} | nDCG@10 = {nd:.4f}\n")

            except Exception as e:
                print(f"[ERROR] {method} failed on {dataset}: {e}")
                traceback.print_exc()
                with open(txt_log_path, "a") as f:
                    f.write(f"{timestamp()} | {dataset} | {method} | ERROR: {str(e)}\n")
                continue

    # --- Save final results ---
    df = pd.DataFrame(rows, columns=["Dataset", "Method", "nDCG@10"])
    pivot = df.pivot_table(index="Method", columns="Dataset", values="nDCG@10", aggfunc="mean")
    out_csv = os.path.join(RESULTS_DIR, os.path.basename(args.output))
    pivot.to_csv(out_csv)

    print(f"\nSaved results to {out_csv}", flush=True)
    print(f"[INFO {timestamp()}] Logs saved to: {os.path.abspath(log_path)}", flush=True)
    with pd.option_context('display.max_columns', None, 'display.width', 160):
        print("\n=== Summary (nDCG@10) ===")
        print(pivot.fillna('-'))

    tee.close()


if __name__ == "__main__":
    run()
