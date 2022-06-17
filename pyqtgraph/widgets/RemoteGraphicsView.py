from ..Qt import QT_LIB, QtCore, QtGui, QtWidgets

import atexit
import enum
import mmap
import os
import random
import sys
import tempfile

from .. import Qt
from .. import CONFIG_OPTIONS
from .. import multiprocess as mp
from .GraphicsView import GraphicsView

__all__ = ['RemoteGraphicsView']


def serialize_mouse_enum(*args):
    # PySide6 (opt-in in 6.3.1) and PyQt6
    # - implemented as python enums
    # - can pickle enums and flags
    # - PyQt6 cannot cast to int
    # PyQt5 5.12, PyQt5 5.15, PySide2 5.15, PySide6 can pickle enums but not flags
    # PySide2 5.12 cannot pickle enums nor flags
    # MouseButtons and KeyboardModifiers are flags
    return [x if isinstance(x, enum.Enum) else int(x) for x in args]


class MouseEvent(QtGui.QMouseEvent):
    @staticmethod
    def get_state(obj, picklable=False):
        typ = obj.type()
        if isinstance(typ, int):
            # PyQt6 returns an int here instead of QEvent.Type,
            # but its QtGui.QMouseEvent constructor takes only QEvent.Type.
            # Note however that its QtCore.QEvent constructor accepts both
            # QEvent.Type and int.
            typ = QtCore.QEvent.Type(typ)
        lpos = obj.position() if hasattr(obj, 'position') else obj.localPos()
        gpos = obj.globalPosition() if hasattr(obj, 'globalPosition') else obj.screenPos()
        btn, btns, mods = obj.button(), obj.buttons(), obj.modifiers()
        if picklable:
            typ, btn, btns, mods = serialize_mouse_enum(typ, btn, btns, mods)
        return typ, lpos, gpos, btn, btns, mods

    def __init__(self, rhs):
        super().__init__(*self.get_state(rhs))

    def __getstate__(self):
        return self.get_state(self, picklable=True)

    def __setstate__(self, state):
        typ, lpos, gpos, btn, btns, mods = state
        typ = QtCore.QEvent.Type(typ)
        btn = QtCore.Qt.MouseButton(btn)
        if not isinstance(btns, enum.Enum):
            btns = QtCore.Qt.MouseButtons(btns)
        if not isinstance(mods, enum.Enum):
            mods = QtCore.Qt.KeyboardModifiers(mods)
        super().__init__(typ, lpos, gpos, btn, btns, mods)


class WheelEvent(QtGui.QWheelEvent):
    @staticmethod
    def get_state(obj, picklable=False):
        # {PyQt6, PySide6}      have position()
        # {PyQt5, PySide2} 5.15 have position()
        # {PyQt5, PySide2} 5.15 have posF() (contrary to C++ docs)
        # {PyQt5, PySide2} 5.12 have posF()
        lpos = obj.position() if hasattr(obj, 'position') else obj.posF()
        gpos = obj.globalPosition() if hasattr(obj, 'globalPosition') else obj.globalPosF()
        pixdel, angdel, btns = obj.pixelDelta(), obj.angleDelta(), obj.buttons()
        mods, phase, inverted = obj.modifiers(), obj.phase(), obj.inverted()
        if picklable:
            btns, mods, phase = serialize_mouse_enum(btns, mods, phase)
        return lpos, gpos, pixdel, angdel, btns, mods, phase, inverted

    def __init__(self, rhs):
        items = list(self.get_state(rhs))
        items[1] = items[0]     # gpos = lpos
        super().__init__(*items)

    def __getstate__(self):
        return self.get_state(self, picklable=True)

    def __setstate__(self, state):
        pos, gpos, pixdel, angdel, btns, mods, phase, inverted = state
        if not isinstance(btns, enum.Enum):
            btns = QtCore.Qt.MouseButtons(btns)
        if not isinstance(mods, enum.Enum):
            mods = QtCore.Qt.KeyboardModifiers(mods)
        phase = QtCore.Qt.ScrollPhase(phase)
        super().__init__(pos, gpos, pixdel, angdel, btns, mods, phase, inverted)


class EnterEvent(QtGui.QEnterEvent):
    @staticmethod
    def get_state(obj):
        lpos = obj.position() if hasattr(obj, 'position') else obj.localPos()
        wpos = obj.scenePosition() if hasattr(obj, 'scenePosition') else obj.windowPos()
        gpos = obj.globalPosition() if hasattr(obj, 'globalPosition') else obj.screenPos()
        return lpos, wpos, gpos

    def __init__(self, rhs):
        super().__init__(*self.get_state(rhs))

    def __getstate__(self):
        return self.get_state(self)

    def __setstate__(self, state):
        super().__init__(*state)


class LeaveEvent(QtCore.QEvent):
    @staticmethod
    def get_state(obj, picklable=False):
        typ = obj.type()
        if picklable:
            typ, = serialize_mouse_enum(typ)
        return typ,

    def __init__(self, rhs):
        super().__init__(*self.get_state(rhs))

    def __getstate__(self):
        return self.get_state(self, picklable=True)

    def __setstate__(self, state):
        typ, = state
        typ = QtCore.QEvent.Type(typ)
        super().__init__(typ)


class RemoteGraphicsView(QtWidgets.QWidget):
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
        QtWidgets.QWidget.__init__(self)

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
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
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

    def mousePressEvent(self, ev):
        self._view.mousePressEvent(MouseEvent(ev), _callSync='off')
        ev.accept()
        return super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        self._view.mouseReleaseEvent(MouseEvent(ev), _callSync='off')
        ev.accept()
        return super().mouseReleaseEvent(ev)

    def mouseMoveEvent(self, ev):
        self._view.mouseMoveEvent(MouseEvent(ev), _callSync='off')
        ev.accept()
        return super().mouseMoveEvent(ev)
        
    def wheelEvent(self, ev):
        self._view.wheelEvent(WheelEvent(ev), _callSync='off')
        ev.accept()
        return super().wheelEvent(ev)

    def enterEvent(self, ev):
        self._view.enterEvent(EnterEvent(ev), _callSync='off')
        return super().enterEvent(ev)
        
    def leaveEvent(self, ev):
        self._view.leaveEvent(LeaveEvent(ev), _callSync='off')
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

            # see functions.py::ndarray_to_qimage() for rationale
            if QT_LIB.startswith('PyQt'):
                # PyQt5, PyQt6 >= 6.0.1
                img_ptr = int(Qt.sip.voidptr(self.shm))
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
