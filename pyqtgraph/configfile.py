"""
configfile.py - Human-readable text configuration file library 
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more information.

Used for reading and writing dictionary objects to a python-like configuration
file format. Data structures may be nested and contain any data type as long
as it can be converted to/from a string using repr and eval.
"""

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

GLOBAL_PATH = None # so not thread safe.


class ParseError(Exception):
    def __init__(self, message, lineNum, line, fileName=None):
        self.lineNum = lineNum
        self.line = line
        self.message = message
        self.fileName = fileName
        Exception.__init__(self, message)
        
    def __str__(self):
        if self.fileName is None:
            msg = "Error parsing string at line %d:\n" % self.lineNum
        else:
            msg = "Error parsing config file '%s' at line %d:\n" % (self.fileName, self.lineNum)
        msg += "%s\n%s" % (self.line, Exception.__str__(self))
        return msg
        

def writeConfigFile(data, fname):
    s = genString(data)
    with open(fname, 'wt') as fd:
        fd.write(s)


def readConfigFile(fname, **scope):
    #cwd = os.getcwd()
    global GLOBAL_PATH
    if GLOBAL_PATH is not None:
        fname2 = os.path.join(GLOBAL_PATH, fname)
        if os.path.exists(fname2):
            fname = fname2

    GLOBAL_PATH = os.path.dirname(os.path.abspath(fname))

    local = {**scope, **units.allUnits}
    local['OrderedDict'] = OrderedDict
    local['readConfigFile'] = readConfigFile
    local['Point'] = Point
    local['QtCore'] = QtCore
    local['ColorMap'] = ColorMap
    local['datetime'] = datetime
    # Needed for reconstructing numpy arrays
    local['array'] = numpy.array
    for dtype in ['int8', 'uint8',
                  'int16', 'uint16', 'float16',
                  'int32', 'uint32', 'float32',
                  'int64', 'uint64', 'float64']:
        local[dtype] = getattr(numpy, dtype)
        
    try:
        #os.chdir(newDir)  ## bad.
        with open(fname, "rt") as fd:
            s = fd.read()
        s = s.replace("\r\n", "\n")
        s = s.replace("\r", "\n")
        data = parseString(s, **local)[1]
    except ParseError:
        sys.exc_info()[1].fileName = fname
        raise
    except:
        print("Error while reading config file %s:"% fname)
        raise
    #finally:
        #os.chdir(cwd)
    return data

def appendConfigFile(data, fname):
    s = genString(data)
    with open(fname, 'at') as fd:
        fd.write(s)


def genString(data, indent=''):
    s = ''
    for k in data:
        sk = str(k)
        if len(sk) == 0:
            print(data)
            raise Exception('blank dict keys not allowed (see data above)')
        if sk[0] == ' ' or ':' in sk:
            print(data)
            raise Exception('dict keys must not contain ":" or start with spaces [offending key is "%s"]' % sk)
        if isinstance(data[k], dict):
            s += indent + sk + ':\n'
            s += genString(data[k], indent + '    ')
        else:
            s += indent + sk + ': ' + repr(data[k]).replace("\n", "\\\n") + '\n'
    return s
    
def parseString(lines, start=0, **scope):
    
    data = OrderedDict()
    if isinstance(lines, str):
        lines = lines.replace("\\\n", "")
        lines = lines.split('\n')
        lines = [l for l in lines if re.search(r'\S', l) and not re.match(r'\s*#', l)]  ## remove empty lines
        
    indent = measureIndent(lines[start])
    ln = start - 1
    
    try:
        while True:
            ln += 1
            #print ln
            if ln >= len(lines):
                break
            
            l = lines[ln]
            
            ## Skip blank lines or lines starting with #
            if re.match(r'\s*#', l) or not re.search(r'\S', l):
                continue
            
            ## Measure line indentation, make sure it is correct for this level
            lineInd = measureIndent(l)
            if lineInd < indent:
                ln -= 1
                break
            if lineInd > indent:
                #print lineInd, indent
                raise ParseError('Indentation is incorrect. Expected %d, got %d' % (indent, lineInd), ln+1, l)
            
            
            if ':' not in l:
                raise ParseError('Missing colon', ln+1, l)
            
            (k, p, v) = l.partition(':')
            k = k.strip()
            v = v.strip()
            
            ## set up local variables to use for eval
            if len(k) < 1:
                raise ParseError('Missing name preceding colon', ln+1, l)
            if k[0] == '(' and k[-1] == ')':  ## If the key looks like a tuple, try evaluating it.
                try:
                    k1 = eval(k, scope)
                    if type(k1) is tuple:
                        k = k1
                except:
                    # If tuple conversion fails, keep the string
                    pass
            if re.search(r'\S', v) and v[0] != '#':  ## eval the value
                try:
                    val = eval(v, scope)
                except:
                    ex = sys.exc_info()[1]
                    raise ParseError("Error evaluating expression '%s': [%s: %s]" % (v, ex.__class__.__name__, str(ex)), (ln+1), l)
            else:
                if ln+1 >= len(lines) or measureIndent(lines[ln+1]) <= indent:
                    #print "blank dict"
                    val = {}
                else:
                    #print "Going deeper..", ln+1
                    (ln, val) = parseString(lines, start=ln+1, **scope)
            data[k] = val
        #print k, repr(val)
    except ParseError:
        raise
    except:
        ex = sys.exc_info()[1]
        raise ParseError("%s: %s" % (ex.__class__.__name__, str(ex)), ln+1, l)
    #print "Returning shallower..", ln+1
    return (ln, data)
    
def measureIndent(s):
    n = 0
    while n < len(s) and s[n] == ' ':
        n += 1
    return n
