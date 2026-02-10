from beir.retrieval.evaluation import EvaluateRetrieval

def results_to_run(results):
    def _to_float(x):
        if isinstance(x, list) and len(x) > 0:
            return float(x[0])
        try:
            return float(x)
        except Exception:
            return 0.0

    return {qid: {docid: _to_float(score) for (docid, score) in items}
            for qid, items in results.items()}


# def eval_ndcg10(run, qrels):
#     evaluator = EvaluateRetrieval()
#     ndcg, _map, recall, precision = evaluator.evaluate(qrels, run, [10])
#     if isinstance(ndcg, dict):
#         val = ndcg.get(10) or ndcg.get("NDCG@10") or list(ndcg.values())[0]
#     else:
#         val = ndcg
#     return round(float(val), 4)

import pytrec_eval

def stringify_keys(obj):
    """Recursively convert all keys to strings."""
    if isinstance(obj, dict):
        return {str(k): stringify_keys(v) for k, v in obj.items()}
    return obj

def intify_relevance(qrels):
    """Ensure all relevance values are integers."""
    qrels_int = {}
    for qid, docs in qrels.items():
        qrels_int[str(qid)] = {str(docid): int(float(rel)) for docid, rel in docs.items()}
    return qrels_int

def eval_ndcg10(run, qrels):
    # Convert both qrels and run to valid pytrec_eval format
    run = stringify_keys(run)
    qrels = intify_relevance(qrels)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "ndcg_cut.10", "recall.10", "P.10"})
    scores = evaluator.evaluate(run)

    # Aggregate mean NDCG@10
    ndcg_values = [v.get("ndcg_cut_10", 0.0) for v in scores.values()]
    return sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0.0


