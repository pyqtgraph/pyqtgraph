"""
configfile.py - Human-readable text configuration file library 
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more information.

Used for reading and writing dictionary objects to a python-like configuration
file format. Data structures may be nested and contain any data type as long
as it can be converted to/from a string using repr and the restricted
expression evaluator in parseString (literals, arithmetic, and calls to the
allow-listed names; arbitrary code is rejected).
"""


import ast
import contextlib
import datetime
import operator
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


## Operators allowed in config file expressions. These cover the arithmetic
## used by unit expressions (e.g. ``20*pA``), flag combinations (e.g.
## ``QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop``)
## and the numeric forms produced by repr() (negative numbers, complex
## numbers such as ``1+2j``).
_BINOP_FUNCS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.BitAnd: operator.and_,
}

_UNARYOP_FUNCS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
    ast.Invert: operator.invert,
}


def _eval_expression(expr, scope):
    """
    Evaluate a config file expression without running arbitrary code.

    Only a restricted grammar is accepted: literals, names from ``scope``,
    lists/tuples/sets/dicts built from those, unary and binary arithmetic,
    attribute access to names that do not start with an underscore, and
    calls to names from ``scope`` or to classes reached by attribute
    access (e.g. ``datetime.datetime(...)``, ``QtCore.QPointF(...)``).
    Everything else raises ValueError. Attribute access to underscore
    names is rejected so the object graph cannot be escaped (e.g.
    ``().__class__.__base__.__subclasses__()``).
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as ex:
        raise ValueError(f"invalid expression: {expr!r}") from ex
    return _eval_node(tree.body, scope)


def _eval_node(node, scope):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in scope:
            return scope[node.id]
        raise ValueError(f"unknown name: {node.id!r}")
    if isinstance(node, ast.List):
        return [_eval_node(elt, scope) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(elt, scope) for elt in node.elts)
    if isinstance(node, ast.Set):
        return {_eval_node(elt, scope) for elt in node.elts}
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                raise ValueError("dict unpacking is not allowed")
            result[_eval_node(key, scope)] = _eval_node(value, scope)
        return result
    if isinstance(node, ast.UnaryOp):
        func = _UNARYOP_FUNCS.get(type(node.op))
        if func is None:
            raise ValueError(
                f"unsupported unary operator: {node.op.__class__.__name__}"
            )
        return func(_eval_node(node.operand, scope))
    if isinstance(node, ast.BinOp):
        func = _BINOP_FUNCS.get(type(node.op))
        if func is None:
            raise ValueError(
                f"unsupported binary operator: {node.op.__class__.__name__}"
            )
        return func(_eval_node(node.left, scope), _eval_node(node.right, scope))
    if isinstance(node, ast.Attribute):
        if node.attr.startswith("_"):
            raise ValueError(
                f"access to private attribute {node.attr!r} is not allowed"
            )
        return getattr(_eval_node(node.value, scope), node.attr)
    if isinstance(node, ast.Call):
        func = _eval_call_func(node.func, scope)
        args = [_eval_node(arg, scope) for arg in node.args]
        kwargs = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                raise ValueError("keyword unpacking is not allowed")
            kwargs[keyword.arg] = _eval_node(keyword.value, scope)
        return func(*args, **kwargs)
    raise ValueError(f"unsupported expression: {node.__class__.__name__}")


def _eval_call_func(func_node, scope):
    """
    Resolve the function of a call expression.

    Calls are allowed to names from ``scope`` and to classes reached by
    attribute access from such a name. Anything else -- method calls,
    calls on constructed values, calls on call results -- is rejected,
    so a config file can construct values but cannot invoke arbitrary
    methods (e.g. ``QtCore.QProcess.execute(...)``).
    """
    if isinstance(func_node, ast.Name):
        if func_node.id in scope:
            return scope[func_node.id]
        raise ValueError(f"unknown name: {func_node.id!r}")
    if isinstance(func_node, ast.Attribute):
        func = _eval_node(func_node, scope)
        if isinstance(func, type):
            return func
        raise ValueError(
            f"only classes may be called through attribute access: {func_node.attr!r}"
        )
    raise ValueError(f"unsupported call: {func_node.__class__.__name__}")


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
    """
    Read a config file written by writeConfigFile.

    Values are parsed with a restricted expression evaluator (see
    parseString): only literals, arithmetic, names from the scope, and
    calls to allow-listed names and classes are accepted, so loading a
    config file cannot execute arbitrary code.
    """
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
    """
    Parse config file lines into an OrderedDict.

    Keys that look like tuples and all values are evaluated with the
    restricted expression evaluator _eval_expression, using ``scope``
    as the allow-list of names.
    """
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

            ## Evaluate the key/value with the restricted expression evaluator
            if len(k) < 1:
                raise ParseError('Missing name preceding colon', ln + 1, l)
            if k[0] == '(' and k[-1] == ')':  # If the key looks like a tuple, try evaluating it.
                with contextlib.suppress(Exception):  # If tuple conversion fails, keep the string
                    k1 = _eval_expression(k, scope)
                    if type(k1) is tuple:
                        k = k1
            if _line_is_real(v):  # evaluate the value
                try:
                    val = _eval_expression(v, scope)
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
