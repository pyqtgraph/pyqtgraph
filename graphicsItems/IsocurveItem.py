

from GraphicsObject import *
import pyqtgraph.functions as fn
from pyqtgraph.Qt import QtGui


class IsocurveItem(GraphicsObject):
    """
    Item displaying an isocurve of a 2D array.
    
    To align this item correctly with an ImageItem,
    call isocurve.setParentItem(image)
    """
    
    def __init__(self, data, level, pen='w'):
        GraphicsObject.__init__(self)
        
        lines = fn.isocurve(data, level)
        
        self.path = QtGui.QPainterPath()
        self.setPen(pen)
        
        for line in lines:
            self.path.moveTo(*line[0])
            self.path.lineTo(*line[1])
            
    def setPen(self, *args, **kwargs):
        self.pen = fn.mkPen(*args, **kwargs)
        self.update()

    def boundingRect(self):
        return self.path.boundingRect()
    
    def paint(self, p, *args):
        p.setPen(self.pen)
        p.drawPath(self.path)
        