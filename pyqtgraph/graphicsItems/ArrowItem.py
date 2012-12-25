from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.functions as fn
import numpy as np
__all__ = ['ArrowItem']

class ArrowItem(QtGui.QGraphicsPathItem):
    """
    For displaying scale-invariant arrows.
    For arrows pointing to a location on a curve, see CurveArrow
    
    """
    
    
    def __init__(self, **opts):
        QtGui.QGraphicsPathItem.__init__(self, opts.get('parent', None))
        if 'size' in opts:
            opts['headLen'] = opts['size']
        if 'width' in opts:
            opts['headWidth'] = opts['width']
        defOpts = {
            'pxMode': True,
            'angle': -150,   ## If the angle is 0, the arrow points left
            'pos': (0,0),
            'headLen': 20,
            'tipAngle': 25,
            'baseAngle': 0,
            'tailLen': None,
            'tailWidth': 3,
            'pen': (200,200,200),
            'brush': (50,50,200),
        }
        defOpts.update(opts)
        
        self.setStyle(**defOpts)
        
        self.setPen(fn.mkPen(defOpts['pen']))
        self.setBrush(fn.mkBrush(defOpts['brush']))
        
        self.rotate(self.opts['angle'])
        self.moveBy(*self.opts['pos'])
    
    def setStyle(self, **opts):
        self.opts = opts
        
        opt = dict([(k,self.opts[k]) for k in ['headLen', 'tipAngle', 'baseAngle', 'tailLen', 'tailWidth']])
        self.path = fn.makeArrowPath(**opt)
        self.setPath(self.path)
        
        if opts['pxMode']:
            self.setFlags(self.flags() | self.ItemIgnoresTransformations)
        else:
            self.setFlags(self.flags() & ~self.ItemIgnoresTransformations)
        
    def paint(self, p, *args):
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        QtGui.QGraphicsPathItem.paint(self, p, *args)
