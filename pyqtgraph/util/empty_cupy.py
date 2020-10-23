"""
Mock some CuPy functions to just use numpy
"""

import numpy as np


def asnumpy(arr, *args, **kwargs):
    return arr


def get_array_module(*args, **kwargs):
    return np
