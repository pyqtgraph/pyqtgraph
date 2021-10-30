"""
ptime.py -  Precision time function made os-independent (should have been taken care of by python)
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more information.
"""


import sys
import warnings
from time import perf_counter as clock
from time import time as system_time

START_TIME = None
time = None

def winTime():
    """Return the current time in seconds with high precision (windows version, use Manager.time() to stay platform independent)."""
    warnings.warn(
        "'pg.time' will be removed from the library in the first release following January, 2022.",
        DeprecationWarning, stacklevel=2
    )
    return clock() + START_TIME

def unixTime():
    """Return the current time in seconds with high precision (unix version, use Manager.time() to stay platform independent)."""
    warnings.warn(
        "'pg.time' will be removed from the library in the first release following January, 2022.",
        DeprecationWarning, stacklevel=2
    )
    return system_time()


if sys.platform.startswith('win'):
    cstart = clock()  ### Required to start the clock in windows
    START_TIME = system_time() - cstart
    
    time = winTime
else:
    time = unixTime
