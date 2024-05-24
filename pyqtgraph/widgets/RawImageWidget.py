"""
RawImageWidget.py
Copyright 2010-2016 Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more information.
"""

import importlib
import numpy as np

from .. import functions as fn
from .. import functions_qimage
from .. import getConfigOption, getCupy
from ..Qt import QtCore, QtGui, QtWidgets, QT_LIB

try:
    if QT_LIB in ["PyQt5", "PySide2"]:
        QtOpenGL = QtGui
        QtOpenGLWidgets = QtWidgets
    else:
        QtOpenGL = importlib.import_module(f"{QT_LIB}.QtOpenGL")
        QtOpenGLWidgets = importlib.import_module(f"{QT_LIB}.QtOpenGLWidgets")
    HAVE_OPENGL = True
except ModuleNotFoundError:
    HAVE_OPENGL = False

__all__ = ['RawImageWidget']

class RawImageWidget(QtWidgets.QWidget):
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
        QtWidgets.QWidget.__init__(self, parent)
        self.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding))
        self.scaled = scaled
        self.opts = None
        self.image = None
        self._cp = getCupy()

    def setImage(self, img, *args, **kargs):
        """
        img must be ndarray of shape (x,y), (x,y,3), or (x,y,4).
        Extra arguments are sent to functions.makeARGB
        """
        if getConfigOption('imageAxisOrder') == 'col-major':
            img = img.swapaxes(0, 1)
        self.opts = (img, args, kargs)
        self.image = None
        self.update()

    def paintEvent(self, ev):
        if self.opts is None:
            return
        if self.image is None:
            img = self.opts[0]
            xp = self._cp.get_array_module(img) if self._cp else np

            qimage = None
            if (
                not self.opts[1]    # no positional arguments
                and {"levels", "lut"}.issuperset(self.opts[2])  # no kwargs besides levels and lut
            ):
                transparentLocations = None
                if img.dtype.kind == "f" and xp.isnan(img.min()):
                    nanmask = xp.isnan(img)
                    if nanmask.ndim == 3:
                        nanmask = nanmask.any(axis=2)
                    transparentLocations = nanmask.nonzero()

                qimage = functions_qimage.try_make_qimage(
                    img,
                    levels=self.opts[2].get("levels"),
                    lut=self.opts[2].get("lut"),
                    transparentLocations=transparentLocations
                )

            if qimage is None:
                argb, alpha = fn.makeARGB(self.opts[0], *self.opts[1], **self.opts[2])
                if self._cp and self._cp.get_array_module(argb) == self._cp:
                    argb = argb.get()  # transfer GPU data back to the CPU
                qimage = fn.ndarray_to_qimage(argb, QtGui.QImage.Format.Format_ARGB32)

            self.image = qimage
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
    __all__.append('RawImageGLWidget')

    class RawImageGLWidget(QtOpenGLWidgets.QOpenGLWidget):
        """
        Similar to RawImageWidget, but uses a GL widget to do all drawing.
        Performance varies between platforms; see examples/VideoSpeedTest for benchmarking.

        Checks if setConfigOptions(imageAxisOrder='row-major') was set.
        """

        def __init__(self, parent=None, smooth=False):
            super().__init__(parent)
            self.image = None
            self.uploaded = False
            self.smooth = smooth
            self.opts = None

            self.m_texture = QtOpenGL.QOpenGLTexture(QtOpenGL.QOpenGLTexture.Target.Target2D)
            self.m_blitter = None

        def setImage(self, img, *args, **kargs):
            """
            img must be ndarray of shape (x,y), (x,y,3), or (x,y,4).
            Extra arguments are sent to functions.makeARGB
            """
            if getConfigOption('imageAxisOrder') == 'col-major':
                img = img.swapaxes(0, 1)
            self.opts = (img, args, kargs)
            self.image = None
            self.uploaded = False
            self.update()

        def initializeGL(self):
            ctx = self.context()

            # in Python, slot will not get called during application termination
            ctx.aboutToBeDestroyed.connect(self.cleanup)

            profile = QtOpenGL.QOpenGLVersionProfile()
            profile.setVersion(2, 0)
            if QT_LIB == 'PySide2':
                self.glfn = ctx.functions()
            elif QT_LIB == 'PyQt5':
                self.glfn = ctx.versionFunctions(profile)
            else:
                self.glfn = QtOpenGL.QOpenGLVersionFunctionsFactory.get(profile, ctx)

            self.m_blitter = QtOpenGL.QOpenGLTextureBlitter()
            self.m_blitter.create()

        def cleanup(self):
            # explicit call of cleanup() is needed during application termination
            self.makeCurrent()
            self.m_texture.destroy()
            if self.m_blitter is not None:
                self.m_blitter.destroy()
                self.m_blitter = None
            self.doneCurrent()

        def uploadTexture(self):
            h, w = self.image.shape[:2]

            if self.m_texture.isCreated() and (w != self.m_texture.width() or h != self.m_texture.height()):
                self.m_texture.destroy()

            if not self.m_texture.isCreated():
                self.m_texture.setFormat(QtOpenGL.QOpenGLTexture.TextureFormat.RGBAFormat)
                self.m_texture.setSize(w, h)
                self.m_texture.allocateStorage()

            filt = QtOpenGL.QOpenGLTexture.Filter.Linear if self.smooth else QtOpenGL.QOpenGLTexture.Filter.Nearest
            self.m_texture.setMinMagFilters(filt, filt)
            self.m_texture.setWrapMode(QtOpenGL.QOpenGLTexture.WrapMode.ClampToBorder)
            self.m_texture.setData(QtOpenGL.QOpenGLTexture.PixelFormat.RGBA, QtOpenGL.QOpenGLTexture.PixelType.UInt8, self.image)

            self.uploaded = True

        def paintGL(self):
            GL_COLOR_BUFFER_BIT = 0x4000
            self.glfn.glClear(GL_COLOR_BUFFER_BIT)

            if self.image is None:
                if self.opts is None:
                    return
                img, args, kwds = self.opts
                self.image, _ = fn.makeRGBA(img, *args, **kwds)

            if not self.uploaded:
                self.uploadTexture()

            target = QtGui.QMatrix4x4()
            self.m_blitter.bind()
            self.m_blitter.blit(self.m_texture.textureId(), target, QtOpenGL.QOpenGLTextureBlitter.Origin.OriginTopLeft)
            self.m_blitter.release()