"""Structure search utilities using embeddings and FAISS."""

__all__ = [
    "FaissEmbeddingDatabase",
    "EmbeddingDatabase",
]


def __getattr__(name: str):
    """Lazily resolve the FAISS-backed classes.

    Importing these eagerly pulls in ``faiss`` as a side effect of importing
    *any* ``foldmatch.search.*`` submodule (e.g. ``embedding_computer``),
    which loads the FAISS shared library in every process at CLI startup —
    even ranks that never touch the vector DB. Deferring the import here means
    ``faiss`` loads only when one of these names is actually accessed (i.e. in
    the rank-0 database-build path).
    """
    if name == "FaissEmbeddingDatabase":
        from foldmatch.search.faiss_database import FaissEmbeddingDatabase
        return FaissEmbeddingDatabase
    if name == "EmbeddingDatabase":
        from foldmatch.search.embedding_database import EmbeddingDatabase
        return EmbeddingDatabase
    raise AttributeError(f"module 'foldmatch.search' has no attribute {name!r}")
