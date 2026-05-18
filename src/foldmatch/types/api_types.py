from dataclasses import dataclass
from enum import Enum
from typing import IO, Tuple, List, Optional, Literal

from typer import FileText

StreamSrc = FileText | IO
StreamTuple = Tuple[str, StreamSrc, str, str] | Tuple[str, StreamSrc, str] | Tuple[str, str]
FileOrStreamTuple = FileText | StreamTuple

Devices = int | List[int] | Literal["auto"]

EmbeddingPath = str | FileText
OptionalPath = Optional[FileText]


class StructureFormat(str, Enum):
    pdb = "pdb"
    mmcif = "mmcif"
    bciff = "binarycif"

class Granularity(str, Enum):
    chain = "chain"
    assembly = "assembly"

class Accelerator(str, Enum):
    cpu = "cpu"
    gpu = "gpu"
    tpu = "tpu"
    hpu = "hpu"
    auto = "auto"

class Strategy(str, Enum):
    ddp = "ddp"
    auto = "auto"

class SrcLocation(str, Enum):
    file = "file"
    stream = "stream"


class StructureLocation(str, Enum):
    local = "local"
    remote = "remote"


class SrcProteinFrom(str, Enum):
    chain = "chain"
    structure = "structure"


class SrcEsmFrom(str, Enum):
    fasta = "fasta"
    structure = "structure"


class SrcAssemblyFrom(str, Enum):
    assembly = "assembly"
    structure = "structure"


class SrcTensorFrom(str, Enum):
    file = "file"
    structure = "structure"
    parquet = "parquet"

class OutFormat(str, Enum):
    csv = "csv"
    pt = "pt"
    parquet = "parquet"
    json = "json"


class ResEmbeddingFormat(str, Enum):
    pt = "pt"
    csv = "csv"

class LogLevel(str, Enum):
    info = "info"
    warn = "warn"
    debug = "debug"


class IndexType(str, Enum):
    """FAISS index variant used when building an embedding database."""
    auto = "auto"        # IndexFlatIP if N<10k else IndexHNSWFlat (legacy default)
    flat = "flat"        # IndexFlatIP — exact, RAM-resident
    hnsw = "hnsw"        # IndexHNSWFlat — approximate, RAM-resident
    ivf_pq = "ivf_pq"    # OPQ + IVF-PQ with OnDiskInvertedLists — approximate, on-disk


@dataclass
class IndexConfig:
    """Tuning knobs for IndexType. All fields are optional; sensible defaults apply."""
    # IVF-PQ
    nlist: int = 16384            # number of IVF cells
    m: int = 64                   # PQ subquantizers (must divide D)
    nbits: int = 8                # bits per PQ subcode
    nprobe: Optional[int] = None  # search-time cells probed; defaults to nlist // 64
    opq_train_size: int = 1_000_000  # training sample size for OPQ + IVF-PQ
    # HNSW
    hnsw_m: int = 32              # HNSW graph connectivity