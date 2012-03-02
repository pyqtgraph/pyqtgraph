from pyqtgraph.Qt import QtCore, QtGui, QtOpenGL

import pyqtgraph.functions as fn
import numpy as np

class RawImageWidget(QtGui.QWidget):
    """
    Widget optimized for very fast video display. 
    Generally using an ImageItem inside GraphicsView is fast enough,
    but if you need even more performance, this widget is about as fast as it gets (but only in unscaled mode).
    """
    def __init__(self, parent=None, scaled=False):
        """
        Setting scaled=True will cause the entire image to be displayed within the boundaries of the widget. This also greatly reduces the speed at which it will draw frames.
        """
        QtGui.QWidget.__init__(self, parent=None)
        self.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding,QtGui.QSizePolicy.Expanding))
        self.scaled = scaled
        self.opts = None
        self.image = None
    
    def setImage(self, img, *args, **kargs):
        """
        img must be ndarray of shape (x,y), (x,y,3), or (x,y,4).
        Extra arguments are sent to functions.makeARGB
        """
        self.opts = (img, args, kargs)
        self.image = None
        self.update()

    def paintEvent(self, ev):
        if self.opts is None:
            return
        if self.image is None:
            argb, alpha = fn.makeARGB(self.opts[0], *self.opts[1], **self.opts[2])
            self.image = fn.makeQImage(argb, alpha)
            self.opts = ()
        #if self.pixmap is None:
            #self.pixmap = QtGui.QPixmap.fromImage(self.image)
        p = QtGui.QPainter(self)
        if self.scaled:
            rect = self.rect()
            ar = rect.width() / float(rect.height())
            imar = self.image.width() / float(self.image.height())
            if ar > imar:
                rect.setWidth(int(rect.width() * imar/ar))
            else:
                rect.setHeight(int(rect.height() * ar/imar))
                
            p.drawImage(rect, self.image)
        else:
            p.drawImage(QtCore.QPointF(), self.image)
        #p.drawPixmap(self.rect(), self.pixmap)
        p.end()


class RawImageGLWidget(QtOpenGL.QGLWidget):
    """
    Similar to RawImageWidget, but uses a GL widget to do all drawing. 
    Generally this will be about as fast as using GraphicsView + ImageItem,
    but performance may vary on some platforms.
    """
    def __init__(self, parent=None, scaled=False):
        QtOpenGL.QGLWidget.__init__(self, parent=None)
        self.scaled = scaled
        self.image = None
    
    def setImage(self, img):
        self.image = fn.makeQImage(img)
        self.update()

    def paintEvent(self, ev):
        if self.image is None:
            return
        p = QtGui.QPainter(self)
        p.drawImage(self.rect(), self.image)
        p.end()


