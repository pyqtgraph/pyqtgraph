# -*- coding: utf-8 -*-
"""
ptime.py -  Precision time function made os-independent (should have been taken care of by python)
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more information.
"""


import sys

if sys.version_info[0] < 3:
    from time import clock
    from time import time as system_time
else:
    from time import perf_counter as clock
    from time import time as system_time

START_TIME = None
time = None

def winTime():
    """Return the current time in seconds with high precision (windows version, use Manager.time() to stay platform independent)."""
    return clock() - START_TIME
    #return systime.time()

def unixTime():
    """Return the current time in seconds with high precision (unix version, use Manager.time() to stay platform independent)."""
    return system_time()

if sys.platform.startswith('win'):
    cstart = clock()  ### Required to start the clock in windows
    START_TIME = system_time() - cstart
    
    time = winTime
else:
    time = unixTime

