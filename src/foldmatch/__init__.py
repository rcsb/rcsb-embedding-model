import os
import sys

if sys.platform == "darwin":
    # PyPI wheels for faiss-cpu and torch each bundle their own libomp.dylib.
    # With two OpenMP runtimes loaded on macOS, FAISS's parallel sections
    # (batched search, HNSW add, IVF-PQ train) deadlock on pthread_mutex_init.
    # Pinning OMP to one thread keeps FAISS out of the parallel path.
    # Must run BEFORE libomp is loaded — i.e., before any torch/faiss import.
    # Linux installs share one libomp and are unaffected.
    # To override on a system with a properly unified libomp:
    #   export OMP_NUM_THREADS=N   # before invoking Python
    os.environ.setdefault("OMP_NUM_THREADS", "1")

from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("foldmatch")
except PackageNotFoundError:
    __version__ = "0.0.0"


def __getattr__(name: str):
    """Lazy attribute access for heavy modules.

    Importing :class:`FoldMatch` transitively loads torch + ESM + lightning,
    which takes ~1s. Resolving it here on first attribute access means
    `from foldmatch import __version__` (used by every CLI module) doesn't
    pay that cost.
    """
    if name == "FoldMatch":
        from foldmatch.foldmatch import FoldMatch as _FoldMatch
        return _FoldMatch
    raise AttributeError(f"module 'foldmatch' has no attribute {name!r}")


__all__ = ["FoldMatch", "__version__"]
