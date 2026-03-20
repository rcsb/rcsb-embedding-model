import logging
import warnings

from rcsb_embedding_model.types.api_types import LogLevel


def arg_devices(devices):
    if len(devices) == 1:
        return devices[0] if devices[0] == "auto" else int(devices[0])
    return [int(x) for x in devices]


def set_log_level(level: LogLevel):
    if level == 'info':
        warnings.filterwarnings("ignore")
        logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s'
        )
    elif level == 'warning':
        logging.basicConfig(
            level=logging.WARN,
            format='> %(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
    elif level == 'debug':
        logging.basicConfig(
            level=logging.DEBUG,
            format='> %(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )