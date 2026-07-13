"""Command-line entry points for corpus acquisition and indexing.

docqa-ingest fetch --query "semaglutide cardiovascular outcomes" --max-docs 100
docqa-ingest index --corpus data/corpus.jsonl
docqa-ingest index --sample          # bundled demo corpus, no key needed
docqa-ingest stats
"""

import argparse
import json
import sys
from pathlib import Path

from ..embeddings import get_embedder
from ..sampledata import sample_corpus_path
from ..store import get_store
from .pipeline import index_documents, load_corpus_jsonl
from .pubmed import fetch_corpus


def _cmd_fetch(args: argparse.Namespace) -> int:
    print(f"Searching PubMed for: {args.query!r} (max {args.max_docs} docs)")
    docs = fetch_corpus(args.query, max_docs=args.max_docs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for doc in docs:
            record = {"id": doc.id, "title": doc.title, "url": doc.url, "text": doc.text}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(docs)} documents with abstracts to {out}")
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    if args.sample:
        with sample_corpus_path() as path:
            docs = load_corpus_jsonl(path)
        source = "bundled sample corpus"
    else:
        docs = load_corpus_jsonl(args.corpus)
        source = args.corpus
    print(f"Indexing {len(docs)} documents from {source}")

    store = get_store(args.store)
    embedder = get_embedder(args.embeddings)
    print(f"store={store.backend}  embedder={embedder.id}")
    try:
        report = index_documents(
            docs, store, embedder, force=args.force, on_progress=lambda msg: print(f"  {msg}")
        )
        print(
            f"Done: {report.indexed} indexed, {report.skipped} skipped (unchanged), "
            f"{report.chunks} chunks embedded."
        )
        print(f"Corpus now: {store.stats()}")
    finally:
        store.close()
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    store = get_store(args.store)
    try:
        print(json.dumps(store.stats(), indent=2))
    finally:
        store.close()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="docqa-ingest", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download a public corpus from PubMed")
    fetch.add_argument("--query", required=True, help="PubMed search query")
    fetch.add_argument("--max-docs", type=int, default=100)
    fetch.add_argument("--out", default="data/corpus.jsonl")
    fetch.set_defaults(func=_cmd_fetch)

    index = sub.add_parser("index", help="chunk, embed, and upsert a corpus into the store")
    corpus_source = index.add_mutually_exclusive_group(required=True)
    corpus_source.add_argument("--corpus", help="path to a corpus .jsonl (id, title, url, text)")
    corpus_source.add_argument("--sample", action="store_true", help="use the bundled demo corpus")
    index.add_argument("--force", action="store_true", help="re-embed even unchanged documents")
    index.add_argument("--store", choices=["sqlite", "pgvector"], default=None)
    index.add_argument("--embeddings", choices=["hash", "openai"], default=None)
    index.set_defaults(func=_cmd_index)

    stats = sub.add_parser("stats", help="print corpus statistics")
    stats.add_argument("--store", choices=["sqlite", "pgvector"], default=None)
    stats.set_defaults(func=_cmd_stats)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
