"""
configfile.py - Human-readable text configuration file library 
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more information.

Used for reading and writing dictionary objects to a python-like configuration
file format. Data structures may be nested and contain any data type as long
as it can be converted to/from a string using repr and eval.
"""


import contextlib
import datetime
import os
import re
import sys
from collections import OrderedDict

import numpy

from . import units
from .colormap import ColorMap
from .Point import Point
from .Qt import QtCore

GLOBAL_PATH = None  # so not thread safe.


class ParseError(Exception):
    def __init__(self, message, lineNum, line, fileName=None):
        self.lineNum = lineNum
        self.line = line
        self.message = message
        self.fileName = fileName
        Exception.__init__(self, message)

    def __str__(self):
        if self.fileName is None:
            msg = f"Error parsing string at line {self.lineNum:d}:\n"
        else:
            msg = f"Error parsing config file '{self.fileName}' at line {self.lineNum:d}:\n"
        msg += f"{self.line}\n{Exception.__str__(self)}"
        return msg


def writeConfigFile(data, fname):
    s = genString(data)
    with open(fname, 'wt') as fd:
        fd.write(s)


def readConfigFile(fname, **scope):
    global GLOBAL_PATH
    if GLOBAL_PATH is not None:
        fname2 = os.path.join(GLOBAL_PATH, fname)
        if os.path.exists(fname2):
            fname = fname2

    GLOBAL_PATH = os.path.dirname(os.path.abspath(fname))

    local = {
        **scope,
        **units.allUnits,
        'OrderedDict': OrderedDict,
        'readConfigFile': readConfigFile,
        'Point': Point,
        'QtCore': QtCore,
        'ColorMap': ColorMap,
        'datetime': datetime,
        # Needed for reconstructing numpy arrays
        'array': numpy.array,
    }
    for dtype in ['int8', 'uint8',
                  'int16', 'uint16', 'float16',
                  'int32', 'uint32', 'float32',
                  'int64', 'uint64', 'float64']:
        local[dtype] = getattr(numpy, dtype)

    try:
        with open(fname, "rt") as fd:
            s = fd.read()
        s = s.replace("\r\n", "\n")
        s = s.replace("\r", "\n")
        data = parseString(s, **local)[1]
    except ParseError:
        sys.exc_info()[1].fileName = fname
        raise
    except:
        print(f"Error while reading config file {fname}:")
        raise
    return data


def appendConfigFile(data, fname):
    s = genString(data)
    with open(fname, 'at') as fd:
        fd.write(s)


def genString(data, indent=''):
    s = ''
    for k in data:
        sk = str(k)
        if not sk:
            print(data)
            raise ValueError('blank dict keys not allowed (see data above)')
        if sk[0] == ' ' or ':' in sk:
            print(data)
            raise ValueError(
                f'dict keys must not contain ":" or start with spaces [offending key is "{sk}"]'
            )
        if isinstance(data[k], dict):
            s += f"{indent}{sk}:\n"
            s += genString(data[k], f'{indent}    ')
        else:
            line = repr(data[k]).replace("\n", "\\\n")
            s += f"{indent}{sk}: {line}\n"
    return s


def parseString(lines, start=0, **scope):
    data = OrderedDict()
    if isinstance(lines, str):
        lines = lines.replace("\\\n", "")
        lines = lines.split('\n')

    indent = None
    ln = start - 1
    l = ''

    try:
        while True:
            ln += 1
            if ln >= len(lines):
                break

            l = lines[ln]

            ## Skip blank lines or lines starting with #
            if not _line_is_real(l):
                continue

            ## Measure line indentation, make sure it is correct for this level
            lineInd = measureIndent(l)
            if indent is None:
                indent = lineInd
            if lineInd < indent:
                ln -= 1
                break
            if lineInd > indent:
                raise ParseError(f'Indentation is incorrect. Expected {indent:d}, got {lineInd:d}', ln + 1, l)

            if ':' not in l:
                raise ParseError('Missing colon', ln + 1, l)

            k, _, v = l.partition(':')
            k = k.strip()
            v = v.strip()

            ## set up local variables to use for eval
            if len(k) < 1:
                raise ParseError('Missing name preceding colon', ln + 1, l)
            if k[0] == '(' and k[-1] == ')':  # If the key looks like a tuple, try evaluating it.
                with contextlib.suppress(Exception):  # If tuple conversion fails, keep the string
                    k1 = eval(k, scope)
                    if type(k1) is tuple:
                        k = k1
            if _line_is_real(v):  # eval the value
                try:
                    val = eval(v, scope)
                except Exception as ex:
                    raise ParseError(
                        f"Error evaluating expression '{v}': [{ex.__class__.__name__}: {ex}]", ln + 1, l
                    ) from ex
            else:
                next_real_ln = next((i for i in range(ln + 1, len(lines)) if _line_is_real(lines[i])), len(lines))
                if ln + 1 >= len(lines) or measureIndent(lines[next_real_ln]) <= indent:
                    val = {}
                else:
                    ln, val = parseString(lines, start=ln + 1, **scope)
            if k in data:
                raise ParseError(f'Duplicate key: {k}', ln + 1, l)
            data[k] = val
    except ParseError:
        raise
    except Exception as ex:
        raise ParseError(f"{ex.__class__.__name__}: {ex}", ln + 1, l) from ex
    return ln, data


def _line_is_real(line):
    return not re.match(r'\s*#', line) and re.search(r'\S', line)


def measureIndent(s):
    n = 0
    while n < len(s) and s[n] == ' ':
        n += 1
    return n
