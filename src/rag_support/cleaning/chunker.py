"""Split cleaned document text into retrieval-sized, overlapping chunks.

Chunk size/overlap are measured in words rather than model tokens. This is a
deliberate simplification: an exact tokenizer either ties the pipeline to one
model family or needs a downloaded vocab file (network access this project
can't always assume -- see docs/data_pipeline.md), while a word count is a
tokenizer-agnostic proxy that's within ~30% of true token count for English
support text, which is plenty precise for choosing a chunk boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class Chunk:
    index: int
    text: str
    word_count: int


def _split_sentences(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    sentences: list[str] = []
    for paragraph in paragraphs:
        sentences.extend(s for s in _SENTENCE_SPLIT_RE.split(paragraph) if s)
    return sentences or [text]


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Greedily pack sentences into ~chunk_size-word windows with overlap.

    Packing whole sentences (rather than a blind sliding window over raw
    words) avoids splitting mid-sentence, which would otherwise hand the
    embedder and the LLM half-formed thoughts at chunk boundaries.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    sentences = _split_sentences(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_words = 0
    index = 0

    def flush() -> None:
        nonlocal current, current_words, index
        if not current:
            return
        chunk_str = " ".join(current).strip()
        chunks.append(Chunk(index=index, text=chunk_str, word_count=current_words))
        index += 1

    for sentence in sentences:
        sentence_words = len(sentence.split())

        # A single sentence longer than chunk_size on its own still gets its
        # own chunk rather than being silently dropped or truncated.
        if current_words + sentence_words > chunk_size and current:
            flush()
            # Carry the tail of the previous chunk forward for overlap.
            overlap_sentences: list[str] = []
            overlap_words = 0
            for s in reversed(current):
                w = len(s.split())
                if overlap_words + w > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_words += w
            current = overlap_sentences
            current_words = overlap_words

        current.append(sentence)
        current_words += sentence_words

    flush()
    return chunks
