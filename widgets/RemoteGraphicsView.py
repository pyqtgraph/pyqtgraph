from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.multiprocess as mp
import pyqtgraph as pg
import numpy as np
import ctypes, os

__all__ = ['RemoteGraphicsView']

class RemoteGraphicsView(QtGui.QWidget):
    def __init__(self, parent=None, *args, **kwds):
        self._img = None
        self._imgReq = None
        QtGui.QWidget.__init__(self)
        self._proc = mp.QtProcess()
        self.pg = self._proc._import('pyqtgraph')
        rpgRemote = self._proc._import('pyqtgraph.widgets.RemoteGraphicsView')
        self._view = rpgRemote.Renderer(*args, **kwds)
        self._view._setProxyOptions(deferGetattr=True)
        self._view.sceneRendered.connect(mp.proxy(self.remoteSceneChanged))
        
    def scene(self):
        return self._view.scene()
        
    def resizeEvent(self, ev):
        ret = QtGui.QWidget.resizeEvent(self, ev)
        self._view.resize(self.size(), _callSync='off')
        return ret
        
    def remoteSceneChanged(self, data):
        self._img = pg.makeQImage(data, alpha=True)
        self.update()
        
    def paintEvent(self, ev):
        if self._img is None:
            return
        p = QtGui.QPainter(self)
        p.drawImage(self.rect(), self._img, self.rect())
        p.end()

class Renderer(pg.GraphicsView):
    
    sceneRendered = QtCore.Signal(object)
    
    def __init__(self, *args, **kwds):
        pg.GraphicsView.__init__(self, *args, **kwds)
        self.scene().changed.connect(self.update)
        self.img = None
        self.renderTimer = QtCore.QTimer()
        self.renderTimer.timeout.connect(self.renderView)
        self.renderTimer.start(16)
        
    def update(self):
        self.img = None
        return pg.GraphicsView.update(self)
        
    def resize(self, size):
        pg.GraphicsView.resize(self, size)
        self.update()
        
    def renderView(self):
        if self.img is None:
            self.img = QtGui.QImage(self.width(), self.height(), QtGui.QImage.Format_ARGB32)
            self.img.fill(0xffffffff)
            p = QtGui.QPainter(self.img)
            self.render(p, self.viewRect(), self.rect())
            p.end()
            self.data = np.fromstring(ctypes.string_at(int(self.img.bits()), self.img.byteCount()), dtype=np.ubyte).reshape(self.height(), self.width(),4).transpose(1,0,2)
            #self.data = ctypes.string_at(int(self.img.bits()), self.img.byteCount())
            self.sceneRendered.emit(self.data)

