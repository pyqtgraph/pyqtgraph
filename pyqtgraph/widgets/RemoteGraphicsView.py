from ..Qt import QtGui, QtCore, QT_LIB
if QT_LIB.startswith('PyQt'):
    from ..Qt import sip
from .. import multiprocess as mp
from .GraphicsView import GraphicsView
from .. import CONFIG_OPTIONS
import numpy as np
import mmap, tempfile, ctypes, atexit, sys, random

__all__ = ['RemoteGraphicsView']
        

class RemoteGraphicsView(QtGui.QWidget):
    """
    Replacement for GraphicsView that does all scene management and rendering on a remote process,
    while displaying on the local widget.
    
    GraphicsItems must be created by proxy to the remote process.
    
    """
    def __init__(self, parent=None, *args, **kwds):
        """
        The keyword arguments 'useOpenGL' and 'backgound', if specified, are passed to the remote
        GraphicsView.__init__(). All other keyword arguments are passed to multiprocess.QtProcess.__init__().
        """
        self._img = None
        self._imgReq = None
        self._sizeHint = (640,480)  ## no clue why this is needed, but it seems to be the default sizeHint for GraphicsView.
                                    ## without it, the widget will not compete for space against another GraphicsView.
        QtGui.QWidget.__init__(self)

        # separate local keyword arguments from remote.
        remoteKwds = {}
        for kwd in ['useOpenGL', 'background']:
            if kwd in kwds:
                remoteKwds[kwd] = kwds.pop(kwd)

        self._proc = mp.QtProcess(**kwds)
        self.pg = self._proc._import('pyqtgraph')
        self.pg.setConfigOptions(**CONFIG_OPTIONS)
        rpgRemote = self._proc._import('pyqtgraph.widgets.RemoteGraphicsView')
        self._view = rpgRemote.Renderer(*args, **remoteKwds)
        self._view._setProxyOptions(deferGetattr=True)
        
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.shm = None
        shmFileName = self._view.shmFileName()
        if sys.platform.startswith('win'):
            self.shmtag = shmFileName
        else:
            self.shmFile = open(shmFileName, 'r')
        
        self._view.sceneRendered.connect(mp.proxy(self.remoteSceneChanged)) #, callSync='off'))
                                                                            ## Note: we need synchronous signals
                                                                            ## even though there is no return value--
                                                                            ## this informs the renderer that it is 
                                                                            ## safe to begin rendering again. 
        
        for method in ['scene', 'setCentralItem']:
            setattr(self, method, getattr(self._view, method))
        
    def resizeEvent(self, ev):
        ret = super().resizeEvent(ev)
        self._view.resize(self.size(), _callSync='off')
        return ret
        
    def sizeHint(self):
        return QtCore.QSize(*self._sizeHint)
        
    def remoteSceneChanged(self, data):
        w, h, size, newfile = data
        #self._sizeHint = (whint, hhint)
        if self.shm is None or self.shm.size != size:
            if self.shm is not None:
                self.shm.close()
            if sys.platform.startswith('win'):
                self.shmtag = newfile   ## on windows, we create a new tag for every resize
                self.shm = mmap.mmap(-1, size, self.shmtag) ## can't use tmpfile on windows because the file can only be opened once.
            elif sys.platform == 'darwin':
                self.shmFile.close()
                self.shmFile = open(self._view.shmFileName(), 'r')
                self.shm = mmap.mmap(self.shmFile.fileno(), size, mmap.MAP_SHARED, mmap.PROT_READ)
            else:
                self.shm = mmap.mmap(self.shmFile.fileno(), size, mmap.MAP_SHARED, mmap.PROT_READ)
        self.shm.seek(0)
        data = self.shm.read(w*h*4)
        self._img = QtGui.QImage(data, w, h, QtGui.QImage.Format_ARGB32)
        self._img.data = data  # data must be kept alive or PySide 1.2.1 (and probably earlier) will crash.
        self.update()
        
    def paintEvent(self, ev):
        if self._img is None:
            return
        p = QtGui.QPainter(self)
        p.drawImage(self.rect(), self._img, QtCore.QRect(0, 0, self._img.width(), self._img.height()))
        p.end()

    def serialize_mouse_common(self, ev):
        if QT_LIB == 'PyQt6':
            # PyQt6 can pickle MouseButtons and KeyboardModifiers but cannot cast to int
            btns = ev.buttons()
            mods = ev.modifiers()
        else:
            # PyQt5, PySide2, PySide6 cannot pickle MouseButtons and KeyboardModifiers
            btns = int(ev.buttons())
            mods = int(ev.modifiers())
        return (btns, mods)

    def serialize_mouse_event(self, ev):
        # lpos, gpos = ev.localPos(), ev.screenPos()
        # RemoteGraphicsView Renderer assumes to be at (0, 0)
        gpos = lpos = ev.localPos()
        btns, mods = self.serialize_mouse_common(ev)
        return (ev.type(), lpos, gpos, ev.button(), btns, mods)

    def serialize_wheel_event(self, ev):
        # lpos, gpos = ev.position(), globalPosition()
        # RemoteGraphicsView Renderer assumes to be at (0, 0)
        gpos = lpos = ev.position()
        btns, mods = self.serialize_mouse_common(ev)
        return (lpos, gpos, ev.pixelDelta(), ev.angleDelta(), btns, mods, ev.phase(), ev.inverted())

    def mousePressEvent(self, ev):
        self._view.mousePressEvent(self.serialize_mouse_event(ev), _callSync='off')
        ev.accept()
        return super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        self._view.mouseReleaseEvent(self.serialize_mouse_event(ev), _callSync='off')
        ev.accept()
        return super().mouseReleaseEvent(ev)

    def mouseMoveEvent(self, ev):
        self._view.mouseMoveEvent(self.serialize_mouse_event(ev), _callSync='off')
        ev.accept()
        return super().mouseMoveEvent(ev)
        
    def wheelEvent(self, ev):
        self._view.wheelEvent(self.serialize_wheel_event(ev), _callSync='off')
        ev.accept()
        return super().wheelEvent(ev)

    def enterEvent(self, ev):
        lws = ev.localPos(), ev.windowPos(), ev.screenPos()
        self._view.enterEvent(lws, _callSync='off')
        return super().enterEvent(ev)
        
    def leaveEvent(self, ev):
        self._view.leaveEvent(ev.type(), _callSync='off')
        return super().leaveEvent(ev)
        
    def remoteProcess(self):
        """Return the remote process handle. (see multiprocess.remoteproxy.RemoteEventHandler)"""
        return self._proc

    def close(self):
        """Close the remote process. After this call, the widget will no longer be updated."""
        self._proc.close()


class Renderer(GraphicsView):
    ## Created by the remote process to handle render requests
    
    sceneRendered = QtCore.Signal(object)
    
    def __init__(self, *args, **kwds):
        ## Create shared memory for rendered image
        #pg.dbg(namespace={'r': self})
        if sys.platform.startswith('win'):
            self.shmtag = "pyqtgraph_shmem_" + ''.join([chr((random.getrandbits(20)%25) + 97) for i in range(20)])
            self.shm = mmap.mmap(-1, mmap.PAGESIZE, self.shmtag) # use anonymous mmap on windows
        else:
            self.shmFile = tempfile.NamedTemporaryFile(prefix='pyqtgraph_shmem_')
            self.shmFile.write(b'\x00' * (mmap.PAGESIZE+1))
            self.shmFile.flush()
            fd = self.shmFile.fileno()
            self.shm = mmap.mmap(fd, mmap.PAGESIZE, mmap.MAP_SHARED, mmap.PROT_WRITE)
        atexit.register(self.close)
        
        GraphicsView.__init__(self, *args, **kwds)
        self.scene().changed.connect(self.update)
        self.img = None
        self.renderTimer = QtCore.QTimer()
        self.renderTimer.timeout.connect(self.renderView)
        self.renderTimer.start(16)
        
    def close(self):
        self.shm.close()
        if not sys.platform.startswith('win'):
            self.shmFile.close()

    def shmFileName(self):
        if sys.platform.startswith('win'):
            return self.shmtag
        else:
            return self.shmFile.name
        
    def update(self):
        self.img = None
        return super().update()
        
    def resize(self, size):
        oldSize = self.size()
        super().resize(size)
        self.resizeEvent(QtGui.QResizeEvent(size, oldSize))
        self.update()
        
    def renderView(self):
        if self.img is None:
            ## make sure shm is large enough and get its address
            if self.width() == 0 or self.height() == 0:
                return
            size = self.width() * self.height() * 4
            if size > self.shm.size():
                if sys.platform.startswith('win'):
                    ## windows says "WindowsError: [Error 87] the parameter is incorrect" if we try to resize the mmap
                    self.shm.close()
                    ## it also says (sometimes) 'access is denied' if we try to reuse the tag.
                    self.shmtag = "pyqtgraph_shmem_" + ''.join([chr((random.getrandbits(20)%25) + 97) for i in range(20)])
                    self.shm = mmap.mmap(-1, size, self.shmtag)
                elif sys.platform == 'darwin':
                    self.shm.close()
                    self.shmFile.close()
                    self.shmFile = tempfile.NamedTemporaryFile(prefix='pyqtgraph_shmem_')
                    self.shmFile.write(b'\x00' * (size + 1))
                    self.shmFile.flush()
                    self.shm = mmap.mmap(self.shmFile.fileno(), size, mmap.MAP_SHARED, mmap.PROT_WRITE)
                else:
                    self.shm.resize(size)
            
            ## render the scene directly to shared memory
            ctypes_obj = ctypes.c_char.from_buffer(self.shm, 0)
            if QT_LIB.startswith('PySide'):
                # PySide2, PySide6
                img_ptr = ctypes_obj
            else:
                # PyQt5, PyQt6
                img_ptr = sip.voidptr(ctypes.addressof(ctypes_obj))

                if QT_LIB == 'PyQt6':
                    img_ptr.setsize(size)

            self.img = QtGui.QImage(img_ptr, self.width(), self.height(), QtGui.QImage.Format_ARGB32)

            self.img.fill(0xffffffff)
            p = QtGui.QPainter(self.img)
            self.render(p, self.viewRect(), self.rect())
            p.end()
            self.sceneRendered.emit((self.width(), self.height(), self.shm.size(), self.shmFileName()))

    def deserialize_mouse_event(self, mouse_event):
        typ, pos, gpos, btn, btns, mods = mouse_event
        typ = QtCore.QEvent.Type(typ)       # this line needed by PyQt5 only
        btns = QtCore.Qt.MouseButtons(btns)
        mods = QtCore.Qt.KeyboardModifiers(mods)
        return QtGui.QMouseEvent(typ, pos, gpos, btn, btns, mods)

    def deserialize_wheel_event(self, wheel_event):
        pos, gpos, pixelDelta, angleDelta, btns, mods, scrollPhase, inverted = wheel_event
        btns = QtCore.Qt.MouseButtons(btns)
        mods = QtCore.Qt.KeyboardModifiers(mods)
        return QtGui.QWheelEvent(pos, gpos, pixelDelta, angleDelta, btns, mods, scrollPhase, inverted)

    def mousePressEvent(self, mouse_event):
        ev = self.deserialize_mouse_event(mouse_event)
        return super().mousePressEvent(ev)

    def mouseMoveEvent(self, mouse_event):
        ev = self.deserialize_mouse_event(mouse_event)
        return super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, mouse_event):
        ev = self.deserialize_mouse_event(mouse_event)
        return super().mouseReleaseEvent(ev)
    
    def wheelEvent(self, wheel_event):
        ev = self.deserialize_wheel_event(wheel_event)
        return super().wheelEvent(ev)

    def enterEvent(self, lws):
        ev = QtGui.QEnterEvent(*lws)
        return super().enterEvent(ev)

    def leaveEvent(self, typ):
        typ = QtCore.QEvent.Type(typ)   # this line needed by PyQt5 only
        ev = QtCore.QEvent(typ)
        return super().leaveEvent(ev)

