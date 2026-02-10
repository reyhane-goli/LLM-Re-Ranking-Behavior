import json, time
from typing import List, Dict, Tuple, Callable
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from tqdm import tqdm

def _get_text_from_raw(raw: str) -> str:
    try:
        obj = json.loads(raw)
        text = obj.get("text") or obj.get("contents") or ""
        title = obj.get("title") or ""
        if title and text:
            return f"{title}. {text}"
        return title or text or raw
    except Exception:
        return raw

class MonoBERTReRanker:
    def __init__(self, model_name: str, device: str = None, max_len: int = 512):
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_len = max_len

    @torch.no_grad()
    def score(self, query, passages, batch_size=8):
        """
        Compute scores for query-passages pairs robustly.
        """
        import torch
        all_scores = []

        def normalize_to_str(x):
            """Return a clean string for any possible passage structure."""
            if isinstance(x, str):
                return x
            elif isinstance(x, dict):
                # If it's a dict from BEIR, take 'text' or fallback to values
                return str(x.get('text') or x.get('body') or x.get('contents') or list(x.values())[0])
            elif isinstance(x, (tuple, list)):
                # Take the second element if tuple is (docid, text), else first
                if len(x) >= 2 and isinstance(x[1], str):
                    return x[1]
                return str(x[0])
            return str(x)

        passages = [normalize_to_str(p) for p in passages]

        for i in range(0, len(passages), batch_size):
            batch_raw = passages[i:i + batch_size]
            batch = []

            for p in batch_raw:
                if isinstance(p, str):
                    batch.append(p)
                elif isinstance(p, (tuple, list)) and len(p) > 0:
                    batch.append(str(p[0]))
                elif isinstance(p, dict) and 'text' in p:
                    batch.append(str(p['text']))
                else:
                    batch.append(str(p))

            # 🩵 FIX: normalize query if it's a dict
            if isinstance(query, dict):
                query = query.get("title") or query.get("text") or str(list(query.values())[0])


            enc = self.tok(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )


            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            logits = out.logits.squeeze()
            if logits.ndim > 1:
                logits = logits[:, -1]
            all_scores.extend(logits.detach().cpu().tolist())

        return all_scores


class MonoT5ReRanker:
    def __init__(self, model_name: str, device: str = None, max_len: int = 512):
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_len = max_len
        self.true_id = self.tok("true", return_tensors="pt").input_ids[0,0]

    @torch.no_grad()
    def score(self, query: str, passages: List[str], batch_size: int = 4) -> List[float]:
        scores = []
        for i in range(0, len(passages), batch_size):
            batch = passages[i:i+batch_size]
            inputs = [f"Query: {query} Document: {p} Relevant:" for p in batch]
            enc = self.tok(inputs, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
            out = self.model.generate(**enc, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
            step = out.scores[0]
            log_probs = torch.log_softmax(step, dim=-1)
            lp = log_probs[:, self.true_id.to(step.device)]
            scores.extend(lp.detach().float().cpu().tolist())
        return scores

def rerank_with_progress(model_name: str, scorer, bm25_results: Dict[str, List[Tuple[str, float]]], queries: Dict[str, str], doc_fetcher: Callable[[str], str], batch_size: int = 8, colour: str = "blue") -> Dict[str, List[Tuple[str, float]]]:
    print(f"[INFO] Starting {model_name} reranking ({len(bm25_results)} queries × {len(next(iter(bm25_results.values()))) if bm25_results else 0} docs)", flush=True)
    start = time.time()
    run = {}
    iterator = tqdm(bm25_results.items(), total=len(bm25_results), desc=f"[{model_name}] rerank", ncols=100, colour=colour)
    for qid, items in iterator:
        q = queries[qid]
        docids = [docid for docid,_ in items]
        passages = [doc_fetcher(did) for did in docids]
        scores = scorer.score(q, passages, batch_size=batch_size)
        paired = list(zip(docids, scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        run[qid] = paired
    dur = time.time()-start
    print(f"[INFO] {model_name} reranking complete in {int(dur//60)}m{int(dur%60):02d}s.", flush=True)
    return run
