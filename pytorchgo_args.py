# -*- coding: UTF-8 -*-
# File: logger.py


import os
import shutil
import os.path
from termcolor import colored
from tabulate import tabulate
from pytorchgo.utils import logger

__all__ = ["set_args", "get_args", "print_args"]

# logger file and directory:
global ARGS
ARGS = None


def set_args(args):
    global ARGS
    ARGS = args


def get_args():
    global ARGS
    return ARGS


def print_args():
    global ARGS
    data = []
    for key, value in vars(ARGS).items():
        data.append([key, value])
    table = tabulate(data, headers=["name", "value"])
    logger.info(colored("Args details", "cyan") + table)

    return ARGS
