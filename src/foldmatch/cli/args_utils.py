import logging
import os
import warnings

from foldmatch.types.api_types import LogLevel


def arg_devices(devices):
    if len(devices) == 1:
        return devices[0] if devices[0] == "auto" else int(devices[0])
    return [int(x) for x in devices]


class RankFilter(logging.Filter):
    def filter(self, record):
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                record.rank = str(dist.get_rank())
                return True
        except Exception:
            pass
        record.rank = os.environ.get("RANK", "0")
        return True



def set_log_level(level: LogLevel):
    handler = logging.StreamHandler()
    handler.addFilter(RankFilter())

    if level == 'info':
        warnings.filterwarnings("ignore")
        logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
        handler.setFormatter(logging.Formatter('[rank %(rank)s] %(message)s'))
        logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    elif level == 'warning':
        handler.setFormatter(logging.Formatter(
            '> %(asctime)s - [rank %(rank)s] - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        ))
        logging.basicConfig(level=logging.WARN, handlers=[handler], force=True)
    elif level == 'debug':
        handler.setFormatter(logging.Formatter(
            '> %(asctime)s - [rank %(rank)s] - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        ))
        logging.basicConfig(level=logging.DEBUG, handlers=[handler], force=True)