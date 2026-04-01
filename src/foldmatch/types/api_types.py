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


class SrcAssemblyFrom(str, Enum):
    assembly = "assembly"
    structure = "structure"


class SrcTensorFrom(str, Enum):
    file = "file"
    structure = "structure"

class OutFormat(str, Enum):
    separated = "separated"
    grouped = "grouped"

class LogLevel(str, Enum):
    info = "info"
    warn = "warn"
    debug = "debug"