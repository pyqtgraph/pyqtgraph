from ..Qt import QtGui, QtCore
from .. import functions as fn

__all__ = ['PathButton']


class PathButton(QtGui.QPushButton):
    """Simple PushButton extension that paints a QPainterPath centered on its face.
    """
    def __init__(self, parent=None, path=None, pen='default', brush=None, size=(30,30), margin=7):
        QtGui.QPushButton.__init__(self, parent)
        self.margin = margin
        self.path = None
        if pen == 'default':
            pen = 'k'
        self.setPen(pen)
        self.setBrush(brush)
        if path is not None:
            self.setPath(path)
        if size is not None:
            self.setFixedWidth(size[0])
            self.setFixedHeight(size[1])
            
    def setBrush(self, brush):
        self.brush = fn.mkBrush(brush)
        
    def setPen(self, *args, **kwargs):
        self.pen = fn.mkPen(*args, **kwargs)
        
    def setPath(self, path):
        self.path = path
        self.update()
        
    def paintEvent(self, ev):
        QtGui.QPushButton.paintEvent(self, ev)
        margin = self.margin
        geom = QtCore.QRectF(0, 0, self.width(), self.height()).adjusted(margin, margin, -margin, -margin)
        rect = self.path.boundingRect()
        scale = min(geom.width() / float(rect.width()), geom.height() / float(rect.height()))
        
        p = QtGui.QPainter(self)
        p.setRenderHint(p.Antialiasing)
        p.translate(geom.center())
        p.scale(scale, scale)
        p.translate(-rect.center())
        p.setPen(self.pen)
        p.setBrush(self.brush)
        p.drawPath(self.path)
        p.end()

    
