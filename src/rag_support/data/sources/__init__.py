from rag_support.data.sources.base import DataSource, RawRecord
from rag_support.data.sources.huggingface_source import HuggingFaceSource
from rag_support.data.sources.local_sample_source import LocalSampleSource

__all__ = ["DataSource", "RawRecord", "HuggingFaceSource", "LocalSampleSource", "get_data_source"]


def get_data_source(mode: str) -> DataSource:
    """Factory: resolve a `DataSourceMode` value to a concrete `DataSource`.

    Kept as a plain function (not a class registry) since there are only two
    sources today -- if a third shows up, promote this to the same
    provider-registry pattern used in `embeddings/factory.py`.
    """
    if mode == "hf":
        return HuggingFaceSource()
    if mode == "sample":
        return LocalSampleSource()
    raise ValueError(f"Unknown data source mode: {mode!r}")
