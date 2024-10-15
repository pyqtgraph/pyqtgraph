from math import atan2, degrees

import numpy as np

from . import SRTTransform3D
from .Point import Point
from .Qt import QtGui


class SRTTransform(QtGui.QTransform):
    """Transform that can always be represented as a combination of 3 matrices: scale * rotate * translate
    This transform has no shear; angles are always preserved.
    """
    def __init__(self, init=None):
        QtGui.QTransform.__init__(self)
        self.reset()
        
        if init is None:
            return
        elif isinstance(init, dict):
            self.restoreState(init)
        elif isinstance(init, SRTTransform):
            self._state = {
                'pos': Point(init._state['pos']),
                'scale': Point(init._state['scale']),
                'angle': init._state['angle']
            }
            self.update()
        elif isinstance(init, QtGui.QTransform):
            self.setFromQTransform(init)
        elif isinstance(init, QtGui.QMatrix4x4):
            self.setFromMatrix4x4(init)
        else:
            raise Exception("Cannot create SRTTransform from input type: %s" % str(type(init)))

    def getScale(self):
        return self._state['scale']
        
    def getRotation(self):
        return self._state['angle']
        
    def getTranslation(self):
        return self._state['pos']
    
    def reset(self):
        self._state = {
            'pos': Point(0,0),
            'scale': Point(1,1),
            'angle': 0.0  ## in degrees
        }
        self.update()
        
    def setFromQTransform(self, tr):
        p1 = Point(tr.map(0., 0.))
        p2 = Point(tr.map(1., 0.))
        p3 = Point(tr.map(0., 1.))
        
        dp2 = Point(p2-p1)
        dp3 = Point(p3-p1)
        
        ## detect flipped axes
        if dp2.angle(dp3, units="radians") > 0:
            da = 0
            sy = -1.0
        else:
            da = 0
            sy = 1.0
            
        self._state = {
            'pos': Point(p1),
            'scale': Point(dp2.length(), dp3.length() * sy),
            'angle': degrees(atan2(dp2[1], dp2[0])) + da
        }
        self.update()
        
    def setFromMatrix4x4(self, m):
        m = SRTTransform3D.SRTTransform3D(m)
        angle, axis = m.getRotation()
        if angle != 0 and (axis[0] != 0 or axis[1] != 0 or axis[2] != 1):
            print("angle: %s  axis: %s" % (str(angle), str(axis)))
            raise Exception("Can only convert 4x4 matrix to 3x3 if rotation is around Z-axis.")
        self._state = {
            'pos': Point(m.getTranslation()),
            'scale': Point(m.getScale()),
            'angle': angle
        }
        self.update()
        
    def translate(self, *args):
        """Acceptable arguments are: 
           x, y
           [x, y]
           Point(x,y)"""
        t = Point(*args)
        self.setTranslate(self._state['pos']+t)
        
    def setTranslate(self, *args):
        """Acceptable arguments are: 
           x, y
           [x, y]
           Point(x,y)"""
        self._state['pos'] = Point(*args)
        self.update()
        
    def scale(self, *args):
        """Acceptable arguments are: 
           x, y
           [x, y]
           Point(x,y)"""
        s = Point(*args)
        self.setScale(self._state['scale'] * s)
        
    def setScale(self, *args):
        """Acceptable arguments are: 
           x, y
           [x, y]
           Point(x,y)"""
        self._state['scale'] = Point(*args)
        self.update()
        
    def rotate(self, angle):
        """Rotate the transformation by angle (in degrees)"""
        self.setRotate(self._state['angle'] + angle)
        
    def setRotate(self, angle):
        """Set the transformation rotation to angle (in degrees)"""
        self._state['angle'] = angle
        self.update()

    def __truediv__(self, t):
        """A / B  ==  B^-1 * A"""
        dt = t.inverted()[0] * self
        return SRTTransform(dt)
        
    def __div__(self, t):
        return self.__truediv__(t)
        
    def __mul__(self, t):
        return SRTTransform(QtGui.QTransform.__mul__(self, t))

    def saveState(self):
        p = self._state['pos']
        s = self._state['scale']
        return {'pos': (p[0], p[1]), 'scale': (s[0], s[1]), 'angle': self._state['angle']}

    def __reduce__(self):
        return SRTTransform, (self.saveState(),)

    def restoreState(self, state):
        self._state['pos'] = Point(state.get('pos', (0,0)))
        self._state['scale'] = Point(state.get('scale', (1.,1.)))
        self._state['angle'] = state.get('angle', 0)
        self.update()

    def update(self):
        QtGui.QTransform.reset(self)
        ## modifications to the transform are multiplied on the right, so we need to reverse order here.
        QtGui.QTransform.translate(self, *self._state['pos'])
        QtGui.QTransform.rotate(self, self._state['angle'])
        QtGui.QTransform.scale(self, *self._state['scale'])

    def __repr__(self):
        return str(self.saveState())
        
    def matrix(self):
        return np.array([[self.m11(), self.m12(), self.m13()],[self.m21(), self.m22(), self.m23()],[self.m31(), self.m32(), self.m33()]])
