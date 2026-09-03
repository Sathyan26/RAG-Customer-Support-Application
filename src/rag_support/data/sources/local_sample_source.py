"""Offline data source: the bundled, checked-in sample knowledge base.

Zero network calls, zero API keys -- this is what `DATA_SOURCE=sample` (the
default) points at, and what CI and this project's own tests run against.
See `scripts/generate_sample_dataset.py` for how the dataset was built and
why ~15% of it is deliberately messy.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from importlib import resources

from rag_support.data.sources.base import DataSource, RawRecord
from rag_support.logging_config import get_logger

logger = get_logger(__name__)

_DATASET_PACKAGE = "rag_support.data.sample_dataset"
_DATASET_FILE = "support_kb_raw.jsonl"


class LocalSampleSource(DataSource):
    name = "sample"

    def fetch(self) -> Iterator[RawRecord]:
        dataset_path = resources.files(_DATASET_PACKAGE).joinpath(_DATASET_FILE)
        logger.info("Reading bundled sample dataset from %s", dataset_path)

        with dataset_path.open("r", encoding="utf-8") as f:
            count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                yield RawRecord(
                    external_id=row["external_id"],
                    text=row["text"],
                    category=row.get("category"),
                    intent=row.get("intent"),
                    title=row.get("title"),
                    metadata=row.get("metadata", {}),
                )
                count += 1
        logger.info("Sample source yielded %d records", count)
