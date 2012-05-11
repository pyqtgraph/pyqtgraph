from pyqtgraph.Qt import QtGui, QtCore
from .UIGraphicsItem import *
import numpy as np
import pyqtgraph.functions as fn

__all__ = ['ScaleBar']
class ScaleBar(UIGraphicsItem):
    """
    Displays a rectangular bar with 10 divisions to indicate the relative scale of objects on the view.
    """
    def __init__(self, size, width=5, color=(100, 100, 255)):
        UIGraphicsItem.__init__(self)
        self.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        
        self.brush = fn.mkBrush(color)
        self.pen = fn.mkPen((0,0,0))
        self._width = width
        self.size = size
        
    def paint(self, p, opt, widget):
        UIGraphicsItem.paint(self, p, opt, widget)
        
        rect = self.boundingRect()
        unit = self.pixelSize()
        y = rect.bottom() + (rect.top()-rect.bottom()) * 0.02
        y1 = y + unit[1]*self._width
        x = rect.right() + (rect.left()-rect.right()) * 0.02
        x1 = x - self.size
        
        p.setPen(self.pen)
        p.setBrush(self.brush)
        rect = QtCore.QRectF(
            QtCore.QPointF(x1, y1), 
            QtCore.QPointF(x, y)
        )
        p.translate(x1, y1)
        p.scale(rect.width(), rect.height())
        p.drawRect(0, 0, 1, 1)
        
        alpha = np.clip(((self.size/unit[0]) - 40.) * 255. / 80., 0, 255)
        p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, alpha)))
        for i in range(1, 10):
            #x2 = x + (x1-x) * 0.1 * i
            x2 = 0.1 * i
            p.drawLine(QtCore.QPointF(x2, 0), QtCore.QPointF(x2, 1))
        

    def setSize(self, s):
        self.size = s
        
