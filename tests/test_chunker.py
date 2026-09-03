import pytest

from rag_support.cleaning.chunker import chunk_text


def test_short_text_produces_a_single_chunk():
    text = "This is a short sentence. Here is another one."
    chunks = chunk_text(text, chunk_size=800, overlap=120)
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert "short sentence" in chunks[0].text


def test_long_text_is_split_into_multiple_chunks_with_overlap():
    sentence = "The quick brown fox jumps over the lazy dog and keeps running. "
    text = sentence * 60  # ~600 words repeated -> comfortably over chunk_size
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    # Every chunk should respect the target size reasonably closely (a single
    # oversized sentence is the only thing allowed to exceed it).
    for chunk in chunks[:-1]:
        assert chunk.word_count <= 100 + 15  # small slack for sentence granularity

    # Consecutive chunks should share some words (the overlap).
    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    assert first_words & second_words


def test_chunk_indices_are_sequential():
    sentence = "Sentence number filler content here for testing purposes today. "
    text = sentence * 40
    chunks = chunk_text(text, chunk_size=60, overlap=10)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=50, overlap=50)


def test_a_single_oversized_sentence_still_gets_its_own_chunk():
    huge_sentence = "word " * 500 + "."
    chunks = chunk_text(huge_sentence, chunk_size=50, overlap=10)
    assert len(chunks) >= 1
    assert sum(c.word_count for c in chunks) >= 500


def test_empty_text_produces_no_crash():
    chunks = chunk_text("", chunk_size=100, overlap=10)
    # An empty document legitimately produces zero (or one empty) chunks;
    # the important thing is it doesn't raise.
    assert isinstance(chunks, list)
