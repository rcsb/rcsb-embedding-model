from importlib_metadata import version, PackageNotFoundError
from foldmatch.foldmatch import FoldMatch

try:
    __version__ = version("foldmatch")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["FoldMatch", "__version__"]
