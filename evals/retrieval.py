"""No-cost retrieval regression. Invoke as python -m evals.retrieval."""
import argparse
import json
import tempfile
from pathlib import Path
from statistics import mean

from research_agent.config import settings
from research_agent.knowledge import KnowledgeStore


def evaluate(top_k=3):
    dataset = json.loads(Path(__file__).with_name("retrieval_dataset.json").read_text(encoding="utf-8"))
    rows = []
    old_mode = settings.retrieval_mode
    try:
        settings.retrieval_mode = "bm25"
        with tempfile.TemporaryDirectory(prefix="research-eval-") as folder:
            store = KnowledgeStore(str(Path(folder) / "knowledge.db"))
            for document in dataset["documents"]:
                store.add("evaluation", document["name"], document["text"].encode())
            for case in dataset["cases"]:
                results = store.search("evaluation", case["query"], top_k, mode="bm25")
                actual = list(dict.fromkeys(item["title"] for item in results))
                expected = set(case["relevant"])
                hits = expected & set(actual)
                rank = next((i for i, name in enumerate(actual, 1) if name in expected), None)
                rows.append({"query": case["query"], "retrieved": actual,
                    "recall": len(hits) / len(expected) if expected else None,
                    "reciprocal_rank": 1 / rank if rank else 0,
                    "empty_correct": not actual if not expected else None})
    finally:
        settings.retrieval_mode = old_mode
    answerable = [row for row in rows if row["recall"] is not None]
    unanswerable = [row for row in rows if row["empty_correct"] is not None]
    return {"dataset": dataset["description"], "retrieval_mode": "bm25", "top_k": top_k,
            "cases": len(rows), "document_recall_at_k": mean(r["recall"] for r in answerable),
            "mrr_at_k": mean(r["reciprocal_rank"] for r in answerable),
            "empty_query_accuracy": mean(r["empty_correct"] for r in unanswerable),
            "results": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3, choices=range(1, 21))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.top_k)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
