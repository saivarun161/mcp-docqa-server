"""Measure retrieval quality against a labeled testset.

Each testset line is {"question": ..., "expected_doc_id": ...}. For every
question we retrieve top-k chunks, reduce them to a ranked list of distinct
documents, and check where the expected document lands. Reported metrics:

* recall@1 / @3 / @5 — is the right document in the top n?
* MRR — mean reciprocal rank of the right document.

    docqa-eval                     # bundled testset against the configured store
    docqa-eval --testset path.jsonl --min-recall5 0.8   # gate for CI
"""

import argparse
import json
import sys
from pathlib import Path

from ..retriever import Retriever
from ..sampledata import sample_testset_path


def load_testset(path: str | Path) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def ranked_doc_ids(retriever: Retriever, question: str, k: int) -> list[str]:
    """Distinct doc ids in retrieval order (best chunk decides a doc's rank)."""
    seen: list[str] = []
    for result in retriever.search(question, k=k):
        if result.doc_id not in seen:
            seen.append(result.doc_id)
    return seen


def evaluate(retriever: Retriever, cases: list[dict], k: int = 5) -> dict:
    ranks: list[int | None] = []
    rows = []
    for case in cases:
        docs = ranked_doc_ids(retriever, case["question"], k)
        rank = docs.index(case["expected_doc_id"]) + 1 if case["expected_doc_id"] in docs else None
        ranks.append(rank)
        rows.append((case["question"], case["expected_doc_id"], rank))

    def recall_at(n: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= n) / len(ranks)

    mrr = sum(1.0 / r for r in ranks if r is not None) / len(ranks)
    return {
        "n": len(ranks),
        "recall@1": recall_at(1),
        "recall@3": recall_at(3),
        "recall@5": recall_at(5),
        "mrr": mrr,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="docqa-eval", description=__doc__)
    parser.add_argument("--testset", default=None, help="path to testset .jsonl")
    parser.add_argument("--k", type=int, default=5, help="retrieval depth per question")
    parser.add_argument(
        "--min-recall5",
        type=float,
        default=None,
        help="exit non-zero if recall@5 falls below this (CI gate)",
    )
    args = parser.parse_args()

    if args.testset:
        cases = load_testset(args.testset)
    else:
        with sample_testset_path() as path:
            cases = load_testset(path)

    retriever = Retriever()
    report = evaluate(retriever, cases, k=args.k)

    print(f"\nRetrieval eval — {report['n']} questions, k={args.k}")
    print(f"store={retriever.store.backend}  embedder={retriever.embedder.id}\n")
    for question, expected, rank in report["rows"]:
        marker = f"rank {rank}" if rank else "MISS"
        print(f"  [{marker:>6}] {question}  (expects {expected})")
    print(
        f"\nrecall@1={report['recall@1']:.2f}  recall@3={report['recall@3']:.2f}  "
        f"recall@5={report['recall@5']:.2f}  MRR={report['mrr']:.2f}"
    )

    if args.min_recall5 is not None and report["recall@5"] < args.min_recall5:
        print(f"\nFAIL: recall@5 {report['recall@5']:.2f} < required {args.min_recall5:.2f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
