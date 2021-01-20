# -*- coding: utf-8 -*-
"""
RawImageWidget.py
Copyright 2010-2016 Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more infomation.
"""

from .. import getConfigOption, functions as fn, getCupy
from ..Qt import QtCore, QtGui

try:
    from ..Qt import QtWidgets
    from OpenGL.GL import *

    HAVE_OPENGL = True
except (ImportError, AttributeError):
    # Would prefer `except ImportError` here, but some versions of pyopengl generate
    # AttributeError upon import
    HAVE_OPENGL = False


class RawImageWidget(QtGui.QWidget):
    """
    Widget optimized for very fast video display.
    Generally using an ImageItem inside GraphicsView is fast enough.
    On some systems this may provide faster video. See the VideoSpeedTest example for benchmarking.
    """

    def __init__(self, parent=None, scaled=False):
        """
        Setting scaled=True will cause the entire image to be displayed within the boundaries of the widget.
        This also greatly reduces the speed at which it will draw frames.
        """
        QtGui.QWidget.__init__(self, parent)
        self.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding))
        self.scaled = scaled
        self.opts = None
        self.image = None
        self._cp = getCupy()

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
            if self._cp and self._cp.get_array_module(argb) == self._cp:
                argb = argb.get()  # transfer GPU data back to the CPU
            self.image = fn.makeQImage(argb, alpha)
            self.opts = ()
        # if self.pixmap is None:
            # self.pixmap = QtGui.QPixmap.fromImage(self.image)
        p = QtGui.QPainter(self)
        if self.scaled:
            rect = self.rect()
            ar = rect.width() / float(rect.height())
            imar = self.image.width() / float(self.image.height())
            if ar > imar:
                rect.setWidth(int(rect.width() * imar / ar))
            else:
                rect.setHeight(int(rect.height() * ar / imar))

            p.drawImage(rect, self.image)
        else:
            p.drawImage(QtCore.QPointF(), self.image)
        # p.drawPixmap(self.rect(), self.pixmap)
        p.end()


if HAVE_OPENGL:
    class RawImageGLWidget(QtWidgets.QOpenGLWidget):
        """
        Similar to RawImageWidget, but uses a GL widget to do all drawing.
        Perfomance varies between platforms; see examples/VideoSpeedTest for benchmarking.

        Checks if setConfigOptions(imageAxisOrder='row-major') was set.
        """

        def __init__(self, parent=None, scaled=False):
            QtWidgets.QOpenGLWidget.__init__(self, parent)
            self.scaled = scaled
            self.image = None
            self.uploaded = False
            self.smooth = False
            self.opts = None
            self.row_major = getConfigOption('imageAxisOrder') == 'row-major'

        def setImage(self, img, *args, **kargs):
            """
            img must be ndarray of shape (x,y), (x,y,3), or (x,y,4).
            Extra arguments are sent to functions.makeARGB
            """
            self.opts = (img, args, kargs)
            self.image = None
            self.uploaded = False
            self.update()

        def initializeGL(self):
            self.texture = glGenTextures(1)

        def uploadTexture(self):
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            if self.smooth:
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            else:
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
            # glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)

            if self.row_major:
                image = self.image
            else:
                image = self.image.transpose((1, 0, 2))

            # ## Test texture dimensions first
            # shape = self.image.shape
            # glTexImage2D(GL_PROXY_TEXTURE_2D, 0, GL_RGBA, shape[0], shape[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            # if glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH) == 0:
            # raise Exception("OpenGL failed to create 2D texture (%dx%d); too large for this hardware." % shape[:2])

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.shape[1], image.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
            glDisable(GL_TEXTURE_2D)

        def paintGL(self):
            if self.image is None:
                if self.opts is None:
                    return
                img, args, kwds = self.opts
                kwds['useRGBA'] = True
                self.image, alpha = fn.makeARGB(img, *args, **kwds)

            if not self.uploaded:
                self.uploadTexture()

            dpr = self.devicePixelRatio()
            vp = (0, 0, int(self.width() * dpr), int(self.height() * dpr))
            glViewport(*vp)
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            glColor4f(1, 1, 1, 1)

            glBegin(GL_QUADS)
            glTexCoord2f(0, 1)
            glVertex3f(-1, -1, 0)
            glTexCoord2f(1, 1)
            glVertex3f(1, -1, 0)
            glTexCoord2f(1, 0)
            glVertex3f(1, 1, 0)
            glTexCoord2f(0, 0)
            glVertex3f(-1, 1, 0)
            glEnd()
            glDisable(GL_TEXTURE_3D)
