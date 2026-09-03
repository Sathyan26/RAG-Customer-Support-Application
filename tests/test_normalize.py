from rag_support.cleaning.normalize import (
    clean_text,
    collapse_whitespace,
    content_hash,
    fix_shouting,
    is_low_quality,
    normalize_unicode,
    strip_html,
)


def test_strip_html_removes_tags_and_unescapes_entities():
    assert strip_html("<p>Hello &amp; welcome</p>") == "Hello & welcome"


def test_strip_html_is_a_noop_on_plain_text():
    assert strip_html("just plain text") == "just plain text"


def test_collapse_whitespace_squashes_runs_and_trims():
    assert collapse_whitespace("  a   b\n\n\n\nc  ") == "a b\n\nc"


def test_normalize_unicode_folds_compatibility_characters():
    # U+FF21 FULLWIDTH LATIN CAPITAL LETTER A -> "A"
    assert normalize_unicode("ＡＢＣ") == "ABC"


def test_fix_shouting_sentence_cases_all_caps_lines():
    result = fix_shouting("THIS LINE IS SHOUTING LOUDLY\nThis one is normal.")
    assert result.splitlines()[0] == "This line is shouting loudly"
    assert result.splitlines()[1] == "This one is normal."


def test_fix_shouting_leaves_short_or_mixed_lines_alone():
    # Too few letters to judge, and a normal-case sentence -- neither should change.
    assert fix_shouting("OK") == "OK"
    assert fix_shouting("Some Normal Title Case") == "Some Normal Title Case"


def test_clean_text_pipeline_handles_the_kind_of_noise_the_sample_dataset_injects():
    messy = "  <p>Cancelling doesn&#39;t delete   your data.</p>  \n\n\n"
    cleaned = clean_text(messy)
    assert cleaned == "Cancelling doesn't delete your data."


def test_content_hash_is_stable_and_distinguishes_different_text():
    a = content_hash("hello world")
    b = content_hash("hello world")
    c = content_hash("hello mars")
    assert a == b
    assert a != c
    assert len(a) == 64  # sha256 hex digest


def test_content_hash_makes_noisy_duplicates_collapse_after_cleaning():
    original = "Customer: How do I cancel?\nSupport: Go to Settings > Billing."
    noisy = "<p>Customer: How do I cancel?  \nSupport: Go to Settings > Billing.</p>"
    assert content_hash(clean_text(original)) == content_hash(clean_text(noisy))


def test_is_low_quality_flags_near_empty_and_punctuation_only_text():
    assert is_low_quality("") is True
    assert is_low_quality("...") is True
    assert is_low_quality("ok") is True  # below min_chars default
    assert is_low_quality("This is a real, useful support answer.") is False
