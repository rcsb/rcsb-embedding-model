from enum import Enum
from typing import NewType, Union, IO, Tuple, List, Optional

from typer import FileText

StreamSrc = NewType('StreamSrc', Union[FileText, IO])
StreamTuple = NewType('StreamTuple', Union[
    Tuple[str, StreamSrc, str, str],
    Tuple[str, StreamSrc, str],
    Tuple[str, str]
])
FileOrStreamTuple = NewType('FileOrStreamTuple', Union[FileText, StreamTuple])

Devices = NewType('Devices', Union[int, List[int], "auto"])

EmbeddingPath = Union[str, FileText]
OptionalPath = NewType('OptionalPath', Optional[FileText])


class StructureFormat(str, Enum):
    pdb = "pdb"
    mmcif = "mmcif"
    bciff = "binarycif"


class Accelerator(str, Enum):
    cpu = "cpu"
    gpu = "gpu"
    tpu = "tpu"
    hpu = "hpu"
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
