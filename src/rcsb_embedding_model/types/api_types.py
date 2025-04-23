from enum import Enum
from os import PathLike
from typing import NewType, Union, IO, Tuple, List, Optional

StreamSrc = NewType('StreamSrc', Union[PathLike, IO])
StreamTuple = NewType('StreamTuple', Tuple[StreamSrc, str, str])

Devices = NewType('Devices', Union[int, List[int], "auto"])

OptionalPath = NewType('OptionalPath', Optional[PathLike])


class SrcFormat(str, Enum):
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
    local = "local"
    remote = "remote"
