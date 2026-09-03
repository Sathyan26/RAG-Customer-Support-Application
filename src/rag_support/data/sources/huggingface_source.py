"""Production data source: the public Bitext customer-support dataset.

Pulls "bitext/Bitext-customer-support-llm-chatbot-training-dataset" from the
Hugging Face Hub -- a ~27k-row dataset of (instruction, category, intent,
response) tuples covering the same kind of support domain as the bundled
sample, at real-world scale.

This needs outbound network access to huggingface.co, which not every
environment allows (this project was built in a sandboxed CI-like container
that blocks it -- see docs/data_pipeline.md). Point `DATA_SOURCE=hf` at it
whenever you *do* have Hub access, e.g. running locally or in the Docker
image on a normal network.
"""

from __future__ import annotations

from collections.abc import Iterator

from rag_support.data.sources.base import DataSource, RawRecord
from rag_support.logging_config import get_logger

logger = get_logger(__name__)

DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


class HuggingFaceSource(DataSource):
    name = "hf"

    def __init__(self, dataset_name: str = DATASET_NAME, split: str = "train") -> None:
        self.dataset_name = dataset_name
        self.split = split

    def fetch(self) -> Iterator[RawRecord]:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - exercised via error path
            raise RuntimeError(
                "The 'datasets' package is required for DATA_SOURCE=hf. "
                "Install it with: pip install -e '.[datasets]'"
            ) from exc

        logger.info("Streaming %s (%s split) from the Hugging Face Hub", self.dataset_name, self.split)
        try:
            dataset = load_dataset(self.dataset_name, split=self.split, streaming=True)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load {self.dataset_name!r} from the Hugging Face Hub. "
                "This usually means there's no outbound network access to huggingface.co "
                "from this environment. Set DATA_SOURCE=sample to use the bundled offline "
                "dataset instead."
            ) from exc

        count = 0
        for row in dataset:
            instruction = (row.get("instruction") or "").strip()
            response = (row.get("response") or "").strip()
            if not instruction or not response:
                continue
            count += 1
            yield RawRecord(
                external_id=f"bitext-{count:06d}",
                text=f"Customer: {instruction}\nSupport: {response}",
                category=row.get("category"),
                intent=row.get("intent"),
                title=instruction[:200],
                metadata={"flags": row.get("flags")},
            )
        logger.info("Hugging Face source yielded %d records", count)
