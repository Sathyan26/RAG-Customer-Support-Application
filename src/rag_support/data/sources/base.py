"""The `DataSource` contract every ingestion connector implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RawRecord:
    """One unit of support knowledge as it comes out of a data source,
    before cleaning. Deliberately close to the eventual `Document` row, but
    kept as a separate plain dataclass so sources have zero dependency on
    the storage layer / SQLAlchemy."""

    external_id: str
    text: str
    category: str | None = None
    intent: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DataSource(ABC):
    """A connector that yields `RawRecord`s from wherever it gets its data."""

    name: str

    @abstractmethod
    def fetch(self) -> Iterator[RawRecord]:
        """Yield every record available from this source.

        Implementations should stream rather than materialize a huge list in
        memory where the underlying client supports it (the Hugging Face
        `datasets` library does, via streaming mode).
        """
        raise NotImplementedError
