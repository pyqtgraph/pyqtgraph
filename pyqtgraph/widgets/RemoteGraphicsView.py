from ..Qt import QtGui, QtCore, QT_LIB
if QT_LIB.startswith('PyQt'):
    from ..Qt import sip
from .. import multiprocess as mp
from .GraphicsView import GraphicsView
from .. import CONFIG_OPTIONS
import numpy as np
import mmap, tempfile, os, atexit, sys, random

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
        
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QtGui.QSizePolicy.Policy.Expanding, QtGui.QSizePolicy.Policy.Expanding)
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
            else:
                self.shm = mmap.mmap(self.shmFile.fileno(), size, mmap.MAP_SHARED, mmap.PROT_READ)
        self._img = QtGui.QImage(self.shm, w, h, QtGui.QImage.Format.Format_RGB32).copy()
        self.update()
        
    def paintEvent(self, ev):
        if self._img is None:
            return
        p = QtGui.QPainter(self)
        p.drawImage(self.rect(), self._img, self._img.rect())
        p.end()

    def serialize_mouse_enum(self, *args):
        # PyQt6 can pickle enums and flags but cannot cast to int
        # PyQt5 5.12, PyQt5 5.15, PySide2 5.15, PySide6 can pickle enums but not flags
        # PySide2 5.12 cannot pickle enums nor flags
        # MouseButtons and KeyboardModifiers are flags
        if QT_LIB != 'PyQt6':
            args = [int(x) for x in args]
        return args

    def serialize_mouse_event(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        gpos = ev.globalPosition() if hasattr(ev, 'globalPosition') else ev.screenPos()
        typ, btn, btns, mods = self.serialize_mouse_enum(
            ev.type(), ev.button(), ev.buttons(), ev.modifiers())
        return (typ, lpos, gpos, btn, btns, mods)

    def serialize_wheel_event(self, ev):
        # {PyQt6, PySide6}      have position()
        # {PyQt5, PySide2} 5.15 have position()
        # {PyQt5, PySide2} 5.15 have posF() (contrary to C++ docs)
        # {PyQt5, PySide2} 5.12 have posF()
        lpos = ev.position() if hasattr(ev, 'position') else ev.posF()
        # gpos = ev.globalPosition() if hasattr(ev, 'globalPosition') else ev.globalPosF()
        gpos = lpos     # RemoteGraphicsView Renderer assumes to be at (0, 0)
        btns, mods, phase = self.serialize_mouse_enum(ev.buttons(), ev.modifiers(), ev.phase())
        return (lpos, gpos, ev.pixelDelta(), ev.angleDelta(), btns, mods, phase, ev.inverted())

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
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        wpos = ev.scenePosition() if hasattr(ev, 'scenePosition') else ev.windowPos()
        gpos = ev.globalPosition() if hasattr(ev, 'globalPosition') else ev.screenPos()

        lws = lpos, wpos, gpos
        self._view.enterEvent(lws, _callSync='off')
        return super().enterEvent(ev)
        
    def leaveEvent(self, ev):
        typ, = self.serialize_mouse_enum(ev.type())
        self._view.leaveEvent(typ, _callSync='off')
        return super().leaveEvent(ev)
        
    def remoteProcess(self):
        """Return the remote process handle. (see multiprocess.remoteproxy.RemoteEventHandler)"""
        return self._proc

    def close(self):
        """Close the remote process. After this call, the widget will no longer be updated."""
        self._view.sceneRendered.disconnect()
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
            dpr = self.devicePixelRatioF()
            iwidth = int(self.width() * dpr)
            iheight = int(self.height() * dpr)
            size = iwidth * iheight * 4
            if size > self.shm.size():
                if sys.platform.startswith('win'):
                    ## windows says "WindowsError: [Error 87] the parameter is incorrect" if we try to resize the mmap
                    self.shm.close()
                    ## it also says (sometimes) 'access is denied' if we try to reuse the tag.
                    self.shmtag = "pyqtgraph_shmem_" + ''.join([chr((random.getrandbits(20)%25) + 97) for i in range(20)])
                    self.shm = mmap.mmap(-1, size, self.shmtag)
                elif sys.platform == 'darwin':
                    self.shm.close()
                    fd = self.shmFile.fileno()
                    os.ftruncate(fd, size + 1)
                    self.shm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_WRITE)
                else:
                    self.shm.resize(size)
            
            ## render the scene directly to shared memory

            # see functions.py::makeQImage() for rationale
            if QT_LIB.startswith('PyQt'):
                if QtCore.PYQT_VERSION == 0x60000:
                    img_ptr = sip.voidptr(self.shm)
                else:
                    # PyQt5, PyQt6 >= 6.0.1
                    img_ptr = int(sip.voidptr(self.shm))
            else:
                # PySide2, PySide6
                img_ptr = self.shm

            self.img = QtGui.QImage(img_ptr, iwidth, iheight, QtGui.QImage.Format.Format_RGB32)
            self.img.setDevicePixelRatio(dpr)

            self.img.fill(0xffffffff)
            p = QtGui.QPainter(self.img)
            self.render(p, self.viewRect(), self.rect())
            p.end()
            self.sceneRendered.emit((iwidth, iheight, self.shm.size(), self.shmFileName()))

    def deserialize_mouse_event(self, mouse_event):
        typ, pos, gpos, btn, btns, mods = mouse_event
        typ = QtCore.QEvent.Type(typ)
        if QT_LIB != 'PyQt6':
            btn = QtCore.Qt.MouseButton(btn)
            btns = QtCore.Qt.MouseButtons(btns)
            mods = QtCore.Qt.KeyboardModifiers(mods)
        return QtGui.QMouseEvent(typ, pos, gpos, btn, btns, mods)

    def deserialize_wheel_event(self, wheel_event):
        pos, gpos, pixelDelta, angleDelta, btns, mods, phase, inverted = wheel_event
        if QT_LIB != 'PyQt6':
            btns = QtCore.Qt.MouseButtons(btns)
            mods = QtCore.Qt.KeyboardModifiers(mods)
            phase = QtCore.Qt.ScrollPhase(phase)
        return QtGui.QWheelEvent(pos, gpos, pixelDelta, angleDelta, btns, mods, phase, inverted)

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
        typ = QtCore.QEvent.Type(typ)
        ev = QtCore.QEvent(typ)
        return super().leaveEvent(ev)

