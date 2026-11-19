"""Dataset loading. Uses BEIR's SciFact (scientific claim verification) as the benchmark corpus."""

from __future__ import annotations

import csv
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HF_BASE = "https://huggingface.co/datasets"
DEFAULT_DATA_DIR = Path("data")


@dataclass
class Document:
    id: str
    title: str
    text: str

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.text}" if self.title else self.text


@dataclass
class Corpus:
    documents: list[Document]
    queries: dict[str, str] = field(default_factory=dict)
    # query_id -> {doc_id: relevance grade}
    qrels: dict[str, dict[str, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._by_id = {d.id: d for d in self.documents}

    def get(self, doc_id: str) -> Document:
        return self._by_id[doc_id]

    def __len__(self) -> int:
        return len(self.documents)


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (fixed https URLs)


def download_scifact(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Fetch SciFact from the HuggingFace BEIR mirror and convert it to BEIR's jsonl layout."""
    import pyarrow.parquet as pq

    target = data_dir / "scifact"
    qrels_done = all((target / "qrels" / f"{s}.tsv").exists() for s in ("train", "test"))
    if (target / "corpus.jsonl").exists() and qrels_done:
        return target
    print(f"Downloading SciFact to {target} ...")
    raw = data_dir / "_raw"
    _fetch(
        f"{HF_BASE}/BeIR/scifact/resolve/main/corpus/corpus-00000-of-00001.parquet",
        raw / "c.parquet",
    )
    _fetch(
        f"{HF_BASE}/BeIR/scifact/resolve/main/queries/queries-00000-of-00001.parquet",
        raw / "q.parquet",
    )
    for split in ("train", "test"):
        _fetch(
            f"{HF_BASE}/BeIR/scifact-qrels/resolve/main/{split}.tsv",
            target / "qrels" / f"{split}.tsv",
        )
    for src, name in (("c.parquet", "corpus.jsonl"), ("q.parquet", "queries.jsonl")):
        with open(target / name, "w", encoding="utf-8") as f:
            for row in pq.read_table(raw / src).to_pylist():
                f.write(json.dumps(row) + "\n")
    return target


def load_beir(path: Path, split: str = "test") -> Corpus:
    """Load a BEIR-format dataset directory (corpus.jsonl, queries.jsonl, qrels/<split>.tsv)."""
    documents = []
    with open(path / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            documents.append(Document(str(row["_id"]), row.get("title", ""), row["text"]))

    queries_all = {}
    with open(path / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries_all[str(row["_id"])] = row["text"]

    qrels: dict[str, dict[str, int]] = {}
    with open(path / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for qid, did, score in reader:
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)

    queries = {qid: queries_all[qid] for qid in qrels if qid in queries_all}
    return Corpus(documents=documents, queries=queries, qrels=qrels)


def load_scifact(data_dir: Path = DEFAULT_DATA_DIR, split: str = "test") -> Corpus:
    return load_beir(download_scifact(data_dir), split=split)
