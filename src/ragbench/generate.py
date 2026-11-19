"""Grounded answer generation with citations.

Two backends share one interface:
  * ExtractiveAnswerer: no API key needed. Picks the most query-relevant sentences from the
    retrieved passages and cites them. Deterministic and offline, so the demo always works.
  * ClaudeAnswerer: sends numbered passages to Claude and requires [n] citations.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol

from .data import Corpus, Document
from .retrievers import tokenize

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Answer:
    text: str
    citations: list[str]  # doc ids, in citation order
    backend: str


class Answerer(Protocol):
    def answer(self, question: str, docs: list[Document]) -> Answer: ...


class ExtractiveAnswerer:
    backend = "extractive"

    def __init__(self, max_sentences: int = 3) -> None:
        self._max = max_sentences

    def answer(self, question: str, docs: list[Document]) -> Answer:
        q_terms = set(tokenize(question))
        scored = []
        for n, doc in enumerate(docs, start=1):
            for sent in _SENTENCE.split(doc.text):
                terms = set(tokenize(sent))
                if not terms:
                    continue
                overlap = len(q_terms & terms) / (len(q_terms) or 1)
                # Slight preference for higher-ranked passages.
                scored.append((overlap - 0.01 * n, n, sent.strip()))
        scored.sort(key=lambda t: -t[0])
        picked = scored[: self._max]
        if not picked or picked[0][0] <= 0:
            return Answer(
                "I could not find supporting evidence in the retrieved passages.", [], self.backend
            )
        text = " ".join(f"{s} [{n}]" for _, n, s in picked)
        cited = sorted({n for _, n, _ in picked})
        return Answer(text, [docs[n - 1].id for n in cited], self.backend)


class ClaudeAnswerer:
    backend = "claude"

    SYSTEM = (
        "You answer questions using ONLY the numbered passages provided. Cite every claim with "
        "its passage number in square brackets, like [1] or [2][3]. If the passages do not "
        "contain the answer, say so plainly. Be concise."
    )

    def __init__(self, model: str | None = None) -> None:
        import anthropic

        self._client = anthropic.Anthropic()
        self._model = model or os.environ.get("RAGBENCH_MODEL", "claude-haiku-5-5")

    def answer(self, question: str, docs: list[Document]) -> Answer:
        context = "\n\n".join(f"[{i}] {d.full_text}" for i, d in enumerate(docs, start=1))
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=500,
            system=self.SYSTEM,
            messages=[{"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        cited_nums = sorted(
            {int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(docs)}
        )
        return Answer(text, [docs[n - 1].id for n in cited_nums], f"claude:{self._model}")


def default_answerer() -> Answerer:
    """Claude when an API key and SDK are available, otherwise the offline extractive backend."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return ClaudeAnswerer()
        except ImportError:
            pass
    return ExtractiveAnswerer()


def top_docs(corpus: Corpus, doc_ids: list[str]) -> list[Document]:
    return [corpus.get(i) for i in doc_ids]
