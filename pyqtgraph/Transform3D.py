# -*- coding: utf-8 -*-
from .Qt import QtCore, QtGui
from . import functions as fn
from .Vector import Vector
import numpy as np


class Transform3D(QtGui.QMatrix4x4):
    """
    Extension of QMatrix4x4 with some helpful methods added.
    """
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], (list, tuple, np.ndarray)):
            args = [x for y in args[0] for x in y]
            if len(args) != 16:
                raise TypeError("Single argument to Transform3D must have 16 elements.")
        print(args)
        QtGui.QMatrix4x4.__init__(self, *args)
        
    def matrix(self, nd=3):
        if nd == 3:
            return np.array(self.copyDataTo()).reshape(4,4)
        elif nd == 2:
            m = np.array(self.copyDataTo()).reshape(4,4)
            m[2] = m[3]
            m[:,2] = m[:,3]
            return m[:3,:3]
        else:
            raise Exception("Argument 'nd' must be 2 or 3")
        
    def map(self, obj):
        """
        Extends QMatrix4x4.map() to allow mapping (3, ...) arrays of coordinates
        """
        if isinstance(obj, np.ndarray) and obj.shape[0] in (2,3):
            if obj.ndim >= 2:
                return fn.transformCoordinates(self, obj)
            elif obj.ndim == 1:
                v = QtGui.QMatrix4x4.map(self, Vector(obj))
                return np.array([v.x(), v.y(), v.z()])[:obj.shape[0]]
        elif isinstance(obj, (list, tuple)):
            v = QtGui.QMatrix4x4.map(self, Vector(obj))
            return type(obj)([v.x(), v.y(), v.z()])[:len(obj)]
        else:
            return QtGui.QMatrix4x4.map(self, obj)
            
    def inverted(self):
        inv, b = QtGui.QMatrix4x4.inverted(self)
        return Transform3D(inv), b
