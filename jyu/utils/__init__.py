# -*- coding: UTF-8 -*-
from pathlib import Path

ROOT = Path.cwd()
# ROOT_STR = str(ROOT) # ···/flow_identification

import jyu.utils.conf.config as cfg
import jyu.utils.path as ph
import jyu.utils.color_print as cp
from jyu.utils.color_print import colorstr, print_color
import jyu.utils.plot.plot as plot
import jyu.utils.time as tm

import os
import sys
import platform
import logging.config

# PyTorch Multi-GPU DDP Constants
RANK = int(os.getenv("RANK", -1))
LOGGING_NAME = "JiuYu77"
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # environment booleans
VERBOSE = str(os.getenv("YOLO_VERBOSE", True)).lower() == "true"  # global verbose mode

def emojis(string=""):
    """Return platform-dependent emoji-safe version of string."""
    return string.encode().decode("ascii", "ignore") if WINDOWS else string

def set_logging(name=LOGGING_NAME, verbose=True):
    """Sets up logging for the given name with UTF-8 encoding support."""
    level = logging.INFO if verbose and RANK in {-1, 0} else logging.ERROR  # rank in world for Multi-GPU trainings

    # Configure the console (stdout) encoding to UTF-8
    formatter = logging.Formatter("%(message)s")  # Default formatter
    if WINDOWS and sys.stdout.encoding != "utf-8":
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            elif hasattr(sys.stdout, "buffer"):
                import io

                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
            else:
                sys.stdout.encoding = "utf-8"
        except Exception as e:
            print(f"Creating custom formatter for non UTF-8 environments due to {e}")

            class CustomFormatter(logging.Formatter):
                def format(self, record):
                    """Sets up logging with UTF-8 encoding and configurable verbosity."""
                    return emojis(super().format(record))

            formatter = CustomFormatter("%(message)s")  # Use CustomFormatter to eliminate UTF-8 output as last recourse

    # Create and configure the StreamHandler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    logger = logging.getLogger(name)
    logger.propagate = False  # 防止日志重复打印 logger.propagate 布尔标志, 用于指示消息是否传播给父记录器
    if not logger.handlers:  # 防止日志重复打印 logger.propagate 布尔标志, 用于指示消息是否传播给父记录器
        logger.setLevel(level)
        logger.addHandler(stream_handler)
    return logger

# Set logger
LOGGER = set_logging(LOGGING_NAME, verbose=VERBOSE)  # define globally (used in train.py, val.py, predict.py, etc.)
for logger in "sentry_sdk", "urllib3.connectionpool":
    logging.getLogger(logger).setLevel(logging.CRITICAL + 1)

plot.set_matplotlib()

__all__ = [
    "cfg",
    "ph",
    "cp", "colorstr", "print_color",
    'plot',
    'tm',
]
