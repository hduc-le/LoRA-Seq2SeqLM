import logging
from colorlog import ColoredFormatter
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s] %(message)s",
    #    datefmt='%H:%M:%S.%f',
    datefmt=None,
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green,bold",
        "INFOV": "cyan,bold",
        "WARNING": "yellow",
        "ERROR": "red,bold",
        "CRITICAL": "red,bg_white",
    },
    secondary_log_colors={},
    style="%",
)
ch.setFormatter(formatter)

logger = logging.getLogger("rn")
logger.setLevel(logging.DEBUG)
logger.handlers = []  # No duplicated handlers
logger.propagate = False  # workaround for duplicated logs in ipython
logger.addHandler(ch)

DEFAULT_SEED = 42
DEFAULT_INPUT_MODEL = "t5-base"
NEW_LINE = "\n"
SOURCE_PREFIX = "Input:"
TARGET_PREFIX = "Output:"

SOURCE_FORMAT = """{source_prefix}
{source}""".format(source_prefix=SOURCE_PREFIX, source="{source}")

TARGET_FORMAT = """{target_prefix}
{target}""".format(target_prefix=TARGET_PREFIX, target="{target}")