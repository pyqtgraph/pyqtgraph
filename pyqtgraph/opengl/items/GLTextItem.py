from .. GLGraphicsItem import GLGraphicsItem
from ...Qt import QtCore

__all__ = ['GLTextItem']

class GLTextItem(GLGraphicsItem):
    def __init__(self, X=None, Y=None, Z=None, text=None):
        GLGraphicsItem.__init__(self)
        self.text = text
        self.X = X
        self.Y = Y
        self.Z = Z

    def setGLViewWidget(self, GLViewWidget):
        self.GLViewWidget = GLViewWidget
        
    def setText(self, text):
        self.text = text
        self.update()
    
    def setX(self, X):
        self.X = X
        self.update()

    def setY(self, Y):
        self.Y = Y
        self.update()
        
    def setZ(self, Z):
        self.Z = Z
        self.update()        
    
    def paint(self):
        self.GLViewWidget.qglColor(QtCore.Qt.white)
        self.GLViewWidget.renderText(self.X, self.Y, self.Z, self.text)