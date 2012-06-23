from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.multiprocess as mp
import pyqtgraph as pg
import numpy as np
import mmap, tempfile, ctypes, atexit

__all__ = ['RemoteGraphicsView']

class RemoteGraphicsView(QtGui.QWidget):
    """
    Replacement for GraphicsView that does all scene management and rendering on a remote process,
    while displaying on the local widget.
    
    GraphicsItems must be created by proxy to the remote process.
    
    """
    def __init__(self, parent=None, *args, **kwds):
        self._img = None
        self._imgReq = None
        QtGui.QWidget.__init__(self)
        self._proc = mp.QtProcess()
        self.pg = self._proc._import('pyqtgraph')
        rpgRemote = self._proc._import('pyqtgraph.widgets.RemoteGraphicsView')
        self._view = rpgRemote.Renderer(*args, **kwds)
        self._view._setProxyOptions(deferGetattr=True)
        self.setFocusPolicy(self._view.focusPolicy())
        
        shmFileName = self._view.shmFileName()
        self.shmFile = open(shmFileName, 'r')
        self.shm = mmap.mmap(self.shmFile.fileno(), mmap.PAGESIZE, mmap.MAP_SHARED, mmap.PROT_READ)
        
        self._view.sceneRendered.connect(mp.proxy(self.remoteSceneChanged))
        
        for method in ['scene', 'setCentralItem']:
            setattr(self, method, getattr(self._view, method))
        
    def resizeEvent(self, ev):
        ret = QtGui.QWidget.resizeEvent(self, ev)
        self._view.resize(self.size(), _callSync='off')
        return ret
        
    def remoteSceneChanged(self, data):
        w, h, size = data
        if self.shm.size != size:
            self.shm.close()
            self.shm = mmap.mmap(self.shmFile.fileno(), size, mmap.MAP_SHARED, mmap.PROT_READ)
        self.shm.seek(0)
        self._img = QtGui.QImage(self.shm.read(w*h*4), w, h, QtGui.QImage.Format_ARGB32)
        self.update()
        
    def paintEvent(self, ev):
        if self._img is None:
            return
        p = QtGui.QPainter(self)
        p.drawImage(self.rect(), self._img, QtCore.QRect(0, 0, self._img.width(), self._img.height()))
        p.end()
        
    def mousePressEvent(self, ev):
        self._view.mousePressEvent(ev.type(), ev.pos(), ev.globalPos(), ev.button(), int(ev.buttons()), int(ev.modifiers()), _callSync='off')
        ev.accept()
        return QtGui.QWidget.mousePressEvent(self, ev)

    def mouseReleaseEvent(self, ev):
        self._view.mouseReleaseEvent(ev.type(), ev.pos(), ev.globalPos(), ev.button(), int(ev.buttons()), int(ev.modifiers()), _callSync='off')
        ev.accept()
        return QtGui.QWidget.mouseReleaseEvent(self, ev)

    def mouseMoveEvent(self, ev):
        self._view.mouseMoveEvent(ev.type(), ev.pos(), ev.globalPos(), ev.button(), int(ev.buttons()), int(ev.modifiers()), _callSync='off')
        ev.accept()
        return QtGui.QWidget.mouseMoveEvent(self, ev)
        
    def wheelEvent(self, ev):
        self._view.wheelEvent(ev.pos(), ev.globalPos(), ev.delta(), int(ev.buttons()), int(ev.modifiers()), ev.orientation(), _callSync='off')
        ev.accept()
        return QtGui.QWidget.wheelEvent(self, ev)
    
    def keyEvent(self, ev):
        if self._view.keyEvent(ev.type(), int(ev.modifiers()), text, autorep, count):
            ev.accept()
        return QtGui.QWidget.keyEvent(self, ev)
        
    
    
class Renderer(pg.GraphicsView):
    
    sceneRendered = QtCore.Signal(object)
    
    def __init__(self, *args, **kwds):
        ## Create shared memory for rendered image
        #fd = os.open('/tmp/mmaptest', os.O_CREAT | os.O_TRUNC | os.O_RDWR)
        #os.write(fd, '\x00' * mmap.PAGESIZE)
        self.shmFile = tempfile.NamedTemporaryFile(prefix='pyqtgraph_shmem_')
        self.shmFile.write('\x00' * mmap.PAGESIZE)
        #fh.flush()
        fd = self.shmFile.fileno()
        self.shm = mmap.mmap(fd, mmap.PAGESIZE, mmap.MAP_SHARED, mmap.PROT_WRITE)
        atexit.register(self.close)
        
        pg.GraphicsView.__init__(self, *args, **kwds)
        self.scene().changed.connect(self.update)
        self.img = None
        self.renderTimer = QtCore.QTimer()
        self.renderTimer.timeout.connect(self.renderView)
        self.renderTimer.start(16)
        
    def close(self):
        self.shm.close()
        self.shmFile.close()
        
    def shmFileName(self):
        return self.shmFile.name
        
    def update(self):
        self.img = None
        return pg.GraphicsView.update(self)
        
    def resize(self, size):
        oldSize = self.size()
        pg.GraphicsView.resize(self, size)
        self.resizeEvent(QtGui.QResizeEvent(size, oldSize))
        self.update()
        
    def renderView(self):
        if self.img is None:
            ## make sure shm is large enough and get its address
            size = self.width() * self.height() * 4
            if size > self.shm.size():
                self.shm.resize(size)
            address = ctypes.addressof(ctypes.c_char.from_buffer(self.shm, 0))
            
            ## render the scene directly to shared memory
            self.img = QtGui.QImage(address, self.width(), self.height(), QtGui.QImage.Format_ARGB32)
            self.img.fill(0xffffffff)
            p = QtGui.QPainter(self.img)
            self.render(p, self.viewRect(), self.rect())
            p.end()
            self.sceneRendered.emit((self.width(), self.height(), self.shm.size()))

    def mousePressEvent(self, typ, pos, gpos, btn, btns, mods):
        typ = QtCore.QEvent.Type(typ)
        btns = QtCore.Qt.MouseButtons(btns)
        mods = QtCore.Qt.KeyboardModifiers(mods)
        return pg.GraphicsView.mousePressEvent(self, QtGui.QMouseEvent(typ, pos, gpos, btn, btns, mods))

    def mouseMoveEvent(self, typ, pos, gpos, btn, btns, mods):
        typ = QtCore.QEvent.Type(typ)
        btns = QtCore.Qt.MouseButtons(btns)
        mods = QtCore.Qt.KeyboardModifiers(mods)
        return pg.GraphicsView.mouseMoveEvent(self, QtGui.QMouseEvent(typ, pos, gpos, btn, btns, mods))

    def mouseReleaseEvent(self, typ, pos, gpos, btn, btns, mods):
        typ = QtCore.QEvent.Type(typ)
        btns = QtCore.Qt.MouseButtons(btns)
        mods = QtCore.Qt.KeyboardModifiers(mods)
        return pg.GraphicsView.mouseReleaseEvent(self, QtGui.QMouseEvent(typ, pos, gpos, btn, btns, mods))

    def wheelEvent(self, pos, gpos, d, btns, mods, ori):
        btns = QtCore.Qt.MouseButtons(btns)
        mods = QtCore.Qt.KeyboardModifiers(mods)
        return pg.GraphicsView.wheelEvent(self, QtGui.QWheelEvent(pos, gpos, d, btns, mods, ori))

    def keyEvent(self, typ, mods, text, autorep, count):
        typ = QtCore.QEvent.Type(typ)
        mods = QtCore.Qt.KeyboardModifiers(mods)
        pg.GraphicsView.keyEvent(self, QtGui.QKeyEvent(typ, mods, text, autorep, count))
        return ev.accepted()
        
        
        