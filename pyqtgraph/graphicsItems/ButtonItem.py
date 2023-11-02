from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject

__all__ = ['ButtonItem']
class ButtonItem(GraphicsObject):
    """Button graphicsItem displaying an image."""
    
    clicked = QtCore.Signal(object)
    
    def __init__(self, imageFile=None, width=None, parentItem=None, pixmap=None):
        self.enabled = True
        GraphicsObject.__init__(self)
        if imageFile is not None:
            self.setImageFile(imageFile)
        elif pixmap is not None:
            self.setPixmap(pixmap)

        self._width = width
        if self._width is None:
            self._width = self.pixmap.width() / self.pixmap.devicePixelRatio()

        if parentItem is not None:
            self.setParentItem(parentItem)
        self.setOpacity(0.7)
        
    def setImageFile(self, imageFile):        
        self.setPixmap(QtGui.QPixmap(imageFile))
        
    def setPixmap(self, pixmap):
        self.pixmap = pixmap
        self.update()
        
    def mouseClickEvent(self, ev):
        if self.enabled:
            self.clicked.emit(self)
        
    def hoverEvent(self, ev):
        if not self.enabled:
            return
        if ev.isEnter():
            self.setOpacity(1.0)
        elif ev.isExit():
            self.setOpacity(0.7)

    def disable(self):
        self.enabled = False
        self.setOpacity(0.4)
        
    def enable(self):
        self.enabled = True
        self.setOpacity(0.7)
        
    def paint(self, p, *args):
        p.setRenderHint(p.RenderHint.Antialiasing)
        tgtRect = QtCore.QRectF(0, 0, self._width, self._width)
        srcRect = QtCore.QRectF(self.pixmap.rect())
        p.drawPixmap(tgtRect, self.pixmap, srcRect)
        
    def boundingRect(self):
        return QtCore.QRectF(0, 0, self._width, self._width)
        
