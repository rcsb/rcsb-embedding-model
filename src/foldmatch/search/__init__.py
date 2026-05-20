"""Structure search utilities using embeddings and FAISS."""

from foldmatch.search.faiss_database import FaissEmbeddingDatabase
from foldmatch.search.embedding_database import EmbeddingDatabase

__all__ = [
    "FaissEmbeddingDatabase",
    "EmbeddingDatabase"
]
