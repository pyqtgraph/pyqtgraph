import ctypes
import itertools
import numpy as np
from . import QT_LIB, QtCore, QtGui
from . import compat

__all__ = ["get_qpainterpath_element_array"]

if QT_LIB.startswith('PyQt'):
    from . import sip

class QArrayDataQt5(ctypes.Structure):
    _fields_ = [
        ("ref", ctypes.c_int),
        ("size", ctypes.c_int),
        ("alloc", ctypes.c_uint, 31),
        ("offset", ctypes.c_ssize_t),
    ]

class QPainterPathPrivateQt5(ctypes.Structure):
    _fields_ = [
        ("ref", ctypes.c_int),
        ("adata", ctypes.POINTER(QArrayDataQt5)),
    ]

class QArrayDataQt6(ctypes.Structure):
    _fields_ = [
        ("ref", ctypes.c_int),
        ("flags", ctypes.c_uint),
        ("alloc", ctypes.c_ssize_t),
    ]

class QPainterPathPrivateQt6(ctypes.Structure):
    _fields_ = [
        ("ref", ctypes.c_int),
        ("adata", ctypes.POINTER(QArrayDataQt6)),
        ("data", ctypes.c_void_p),
        ("size", ctypes.c_ssize_t),
    ]

def get_qpainterpath_element_array(qpath, nelems=None):
    writable = nelems is not None
    if writable:
        qpath.reserve(nelems)

    itemsize = 24
    dtype = dict(names=['x','y','c'], formats=['f8', 'f8', 'i4'], itemsize=itemsize)

    ptr0 = compat.unwrapinstance(qpath)
    pte_cp = ctypes.c_void_p.from_address(ptr0)
    if not pte_cp:
        return np.zeros(0, dtype=dtype)

    # _cp : ctypes pointer
    # _ci : ctypes instance
    if QT_LIB in ['PyQt5', 'PySide2']:
        PTR = ctypes.POINTER(QPainterPathPrivateQt5)
        pte_ci = ctypes.cast(pte_cp, PTR).contents
        size_ci = pte_ci.adata[0]
        eptr = ctypes.addressof(size_ci) + size_ci.offset
    elif QT_LIB in ['PyQt6', 'PySide6']:
        PTR = ctypes.POINTER(QPainterPathPrivateQt6)
        pte_ci = ctypes.cast(pte_cp, PTR).contents
        size_ci = pte_ci
        eptr = pte_ci.data
    else:
        raise NotImplementedError

    if writable:
        size_ci.size = nelems
    else:
        nelems = size_ci.size

    vp = compat.voidptr(eptr, itemsize*nelems, writable)
    return np.frombuffer(vp, dtype=dtype)

class PrimitiveArray:
    # Note: This class is an internal implementation detail and is not part
    #       of the public API.
    #
    # QPainter has a C++ native API that takes an array of objects:
    #   drawPrimitives(const Primitive *array, int count, ...)
    # where "Primitive" is one of QPointF, QLineF, QRectF, PixmapFragment
    #
    # PySide (with the exception of drawPixmapFragments) and older PyQt
    # require a Python list of "Primitive" instances to be provided to
    # the respective "drawPrimitives" method.
    #
    # This is inefficient because:
    # 1) constructing the Python list involves calling wrapinstance multiple times.
    #    - this is mitigated here by reusing the instance pointers
    # 2) The binding will anyway have to repack the instances into a contiguous array,
    #    in order to call the underlying C++ native API.
    #
    # Newer PyQt provides sip.array, which is more efficient.
    #
    # PySide's drawPixmapFragments() takes an instance to the first item of a
    # C array of PixmapFragment(s) _and_ the length of the array.
    # There is no overload that takes a Python list of PixmapFragment(s).

    def __init__(self, Klass, nfields, *, use_array=None):
        self._Klass = Klass
        self._nfields = nfields
        self._ndarray = None

        if QT_LIB.startswith('PyQt'):
            if use_array is None:
                use_array = (
                    hasattr(sip, 'array') and
                    (
                        (0x60301 <= QtCore.PYQT_VERSION) or
                        (0x50f07 <= QtCore.PYQT_VERSION < 0x60000)
                    )
                )
            self.use_sip_array = use_array
        else:
            self.use_sip_array = False

        if QT_LIB.startswith('PySide'):
            if use_array is None:
                use_array = (
                    Klass is QtGui.QPainter.PixmapFragment
                )
            self.use_ptr_to_array = use_array
        else:
            self.use_ptr_to_array = False

        self.resize(0)

    def resize(self, size):
        if self._ndarray is not None and len(self._ndarray) == size:
            return

        if self.use_sip_array:
            self._objs = sip.array(self._Klass, size)
            vp = sip.voidptr(self._objs, size*self._nfields*8)
            array = np.frombuffer(vp, dtype=np.float64).reshape((-1, self._nfields))
        elif self.use_ptr_to_array:
            array = np.empty((size, self._nfields), dtype=np.float64)
            self._objs = compat.wrapinstance(array.ctypes.data, self._Klass)
        else:
            array = np.empty((size, self._nfields), dtype=np.float64)
            self._objs = list(map(compat.wrapinstance,
                itertools.count(array.ctypes.data, array.strides[0]),
                itertools.repeat(self._Klass, array.shape[0])))

        self._ndarray = array

    def __len__(self):
        return len(self._ndarray)

    def ndarray(self):
        return self._ndarray

    def instances(self):
        return self._objs
