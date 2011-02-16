# -*- coding: utf-8 -*-
"""
functions.py -  Miscellaneous functions with no other home
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more infomation.
"""

colorAbbrev = {
    'b': (0,0,255,255),
    'g': (0,255,0,255),
    'r': (255,0,0,255),
    'c': (0,255,255,255),
    'm': (255,0,255,255),
    'y': (255,255,0,255),
    'k': (0,0,0,255),
    'w': (255,255,255,255),
}


from PyQt4 import QtGui
from numpy import clip, floor, log

## Copied from acq4/lib/util/functions
SI_PREFIXES = u'yzafpnµm kMGTPEZY'
def siScale(x, minVal=1e-25):
    """Return the recommended scale factor and SI prefix string for x."""
    if abs(x) < minVal:
        m = 0
        x = 0
    else:
        m = int(clip(floor(log(abs(x))/log(1000)), -9.0, 9.0))
    if m == 0:
        pref = ''
    elif m < -8 or m > 8:
        pref = 'e%d' % (m*3)
    else:
        pref = SI_PREFIXES[m+8]
    p = .001**m
    return (p, pref)

def mkBrush(color):
    return QtGui.QBrush(mkColor(color))

def mkPen(arg=None, color=None, width=1, style=None, cosmetic=True, hsv=None, ):
    """Convenience function for making pens. Examples:
    mkPen(color)
    mkPen(color, width=2)
    mkPen(cosmetic=False, width=4.5, color='r')
    mkPen({'color': "FF0", width: 2})
    """
    if isinstance(arg, dict):
        return mkPen(**arg)
    elif arg is not None:
        if isinstance(arg, QtGui.QPen):
            return arg
        color = arg
        
    if color is None:
        color = mkColor(200, 200, 200)
    if hsv is not None:
        color = hsvColor(*hsv)
    else:
        color = mkColor(color)
        
    pen = QtGui.QPen(QtGui.QBrush(color), width)
    pen.setCosmetic(cosmetic)
    if style is not None:
        pen.setStyle(style)
    return pen

def hsvColor(h, s=1.0, v=1.0, a=1.0):
    c = QtGui.QColor()
    c.setHsvF(h, s, v, a)
    return c

def mkColor(*args):
    """make a QColor from a variety of argument types
    accepted types are:
    r, g, b, [a]
    (r, g, b, [a])
    float (greyscale, 0.0-1.0)
    int  (uses intColor)
    (int, hues)  (uses intColor)
    QColor
    "c"    (see colorAbbrev dictionary)
    "RGB"  (strings may optionally begin with "#")
    "RGBA"
    "RRGGBB"
    "RRGGBBAA"
    """
    err = 'Not sure how to make a color from "%s"' % str(args)
    if len(args) == 1:
        if isinstance(args[0], QtGui.QColor):
            return QtGui.QColor(args[0])
        elif isinstance(args[0], float):
            r = g = b = int(args[0] * 255)
            a = 255
        elif isinstance(args[0], basestring):
            c = args[0]
            if c[0] == '#':
                c = c[1:]
            if len(c) == 1:
                (r, g, b, a) = colorAbbrev[c]
            if len(c) == 3:
                r = int(c[0]*2, 16)
                g = int(c[1]*2, 16)
                b = int(c[2]*2, 16)
                a = 255
            elif len(c) == 4:
                r = int(c[0]*2, 16)
                g = int(c[1]*2, 16)
                b = int(c[2]*2, 16)
                a = int(c[3]*2, 16)
            elif len(c) == 6:
                r = int(c[0:2], 16)
                g = int(c[2:4], 16)
                b = int(c[4:6], 16)
                a = 255
            elif len(c) == 8:
                r = int(c[0:2], 16)
                g = int(c[2:4], 16)
                b = int(c[4:6], 16)
                a = int(c[6:8], 16)
        elif hasattr(args[0], '__len__'):
            if len(args[0]) == 3:
                (r, g, b) = args[0]
                a = 255
            elif len(args[0]) == 4:
                (r, g, b, a) = args[0]
            elif len(args[0]) == 2:
                return intColor(*args[0])
            else:
                raise Exception(err)
        elif type(args[0]) == int:
            return intColor(args[0])
        else:
            raise Exception(err)
    elif len(args) == 3:
        (r, g, b) = args
        a = 255
    elif len(args) == 4:
        (r, g, b, a) = args
    else:
        raise Exception(err)
    return QtGui.QColor(r, g, b, a)
    
def colorTuple(c):
    return (c.red(), c.blue(), c.green(), c.alpha())

def colorStr(c):
    """Generate a hex string code from a QColor"""
    return ('%02x'*4) % colorTuple(c)

def intColor(index, hues=9, values=3, maxValue=255, minValue=150, maxHue=360, minHue=0, sat=255):
    """Creates a QColor from a single index. Useful for stepping through a predefined list of colors.
     - The argument "index" determines which color from the set will be returned
     - All other arguments determine what the set of predefined colors will be
     
    Colors are chosen by cycling across hues while varying the value (brightness). By default, there
    are 9 hues and 3 values for a total of 27 different colors. """
    hues = int(hues)
    values = int(values)
    ind = int(index) % (hues * values)
    indh = ind % hues
    indv = ind / hues
    v = minValue + indv * ((maxValue-minValue) / (values-1))
    h = minHue + (indh * (maxHue-minHue)) / hues
    
    c = QtGui.QColor()
    c.setHsv(h, sat, v)
    return c