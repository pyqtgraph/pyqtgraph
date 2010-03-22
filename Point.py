# -*- coding: utf-8 -*-
"""
Point.py -  Extension of QPointF which adds a few missing methods.
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more infomation.
"""

from PyQt4 import QtCore
from math import acos

def clip(x, mn, mx):
    if x > mx:
        return mx
    if x < mn:
        return mn
    return x

class Point(QtCore.QPointF):
    """Extension of QPointF which adds a few missing methods."""
    
    def __init__(self, *args):
        if len(args) == 1:
            if hasattr(args[0], '__getitem__'):
                QtCore.QPointF.__init__(self, float(args[0][0]), float(args[0][1]))
                return
            elif type(args[0]) in [float, int]:
                QtCore.QPointF.__init__(self, float(args[0]), float(args[0]))
                return
        elif len(args) == 2:
            QtCore.QPointF.__init__(self, args[0], args[1])
            return
        QtCore.QPointF.__init__(self, *args)
        
    def __getitem__(self, i):
        if i == 0:
            return self.x()
        elif i == 1:
            return self.y()
        else:
            raise IndexError("Point has no index %d" % i)
        
    def __setitem__(self, i, x):
        if i == 0:
            return self.setX(x)
        elif i == 1:
            return self.setY(x)
        else:
            raise IndexError("Point has no index %d" % i)
        
    def __radd__(self, a):
        return self._math_('__radd__', a)
    
    def __add__(self, a):
        return self._math_('__add__', a)
    
    def __rsub__(self, a):
        return self._math_('__rsub__', a)
    
    def __sub__(self, a):
        return self._math_('__sub__', a)
    
    def __rmul__(self, a):
        return self._math_('__rmul__', a)
    
    def __mul__(self, a):
        return self._math_('__mul__', a)
    
    def __rdiv__(self, a):
        return self._math_('__rdiv__', a)
    
    def __div__(self, a):
        return self._math_('__div__', a)
    
    def __rpow__(self, a):
        return self._math_('__rpow__', a)
    
    def __pow__(self, a):
        return self._math_('__pow__', a)
    
    def _math_(self, op, x):
        #print "point math:", op
        try:
            return Point(getattr(QtCore.QPointF, op)(self, x))
        except:
            x = Point(x)
            return Point(getattr(self[0], op)(x[0]), getattr(self[1], op)(x[1]))
    
    def length(self):
        return (self[0]**2 + self[1]**2) ** 0.5
    
    def angle(self, a):
        n1 = self.length()
        n2 = a.length()
        if n1 == 0. or n2 == 0.:
            return None
        ang = acos(clip(self.dot(a) / (n1 * n2), -1.0, 1.0))
        c = self.cross(a)
        if c > 0:
            ang *= -1.
        return ang
    
    def dot(self, a):
        a = Point(a)
        return self[0]*a[0] + self[1]*a[1]
    
    def cross(self, a):
        a = Point(a)
        return self[0]*a[1] - self[1]*a[0]
    
    def __repr__(self):
        return "Point(%f, %f)" % (self[0], self[1])
    
    
    def min(self):
        return min(self[0], self[1])
    
    def max(self):
        return max(self[0], self[1])