import os
from urllib.parse import urlparse
from rcsb_embedding_model.types.api_types import StructureLocation


def get_structure_location(s: str) -> str:
    # First, attempt to parse as URL
    parsed = urlparse(s)
    if parsed.scheme.lower() in {'http', 'https', 'ftp'} and parsed.netloc:
        return StructureLocation.remote

    # Next, test for an existing file or directory
    if os.path.exists(s):
        return StructureLocation.local

    # Neither URL nor existing file
    raise ValueError(f"Structure file source is neither a recognized URL nor file: {s!r}")