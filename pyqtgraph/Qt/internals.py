import ctypes
import itertools

import numpy as np

from . import QT_LIB, QtCore, QtGui, compat

__all__ = ["get_qpainterpath_element_array"]

if QT_LIB.startswith('PyQt'):
    from . import sip
elif QT_LIB == 'PySide2':
    from PySide2 import __version_info__ as pyside_version_info
elif QT_LIB == 'PySide6':
    from PySide6 import __version_info__ as pyside_version_info

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
        self._capa = -1

        self.use_sip_array = False
        self.use_ptr_to_array = False

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

        if QT_LIB.startswith('PySide'):
            if use_array is None:
                use_array = (
                    Klass is QtGui.QPainter.PixmapFragment
                    or pyside_version_info >= (6, 4, 3)
                )
            self.use_ptr_to_array = use_array

        self.resize(0)

    def resize(self, size):
        if self.use_sip_array:
            # For reference, SIP_VERSION 6.7.8 first arrived
            # in PyQt5_sip 12.11.2 and PyQt6_sip 13.4.2
            if sip.SIP_VERSION >= 0x60708:
                if size <= self._capa:
                    self._size = size
                    return
            else:
                # sip.array prior to SIP_VERSION 6.7.8 had a
                # buggy slicing implementation.
                # so trigger a reallocate for any different size
                if size == self._capa:
                    return

            self._siparray = sip.array(self._Klass, size)

        else:
            if size <= self._capa:
                self._size = size
                return
            self._ndarray = np.empty((size, self._nfields), dtype=np.float64)

            if self.use_ptr_to_array:
                # defer creation
                self._objs = None
            else:
                self._objs = self._wrap_instances(self._ndarray)

        self._capa = size
        self._size = size

    def _wrap_instances(self, array):
        return list(map(compat.wrapinstance,
            itertools.count(array.ctypes.data, array.strides[0]),
            itertools.repeat(self._Klass, array.shape[0])))

    def __len__(self):
        return self._size

    def ndarray(self):
        # ndarray views are cheap to recreate each time
        if self.use_sip_array:
            if (
                sip.SIP_VERSION >= 0x60708 and 
                np.__version__ != "1.22.4"  # TODO: remove me after numpy 1.23+
            ):  # workaround for numpy/sip compatability issue
                # see https://github.com/numpy/numpy/issues/21612
                mv = self._siparray
            else:
                # sip.array prior to SIP_VERSION 6.7.8 had a buggy buffer protocol
                # that set the wrong size.
                # workaround it by going through a sip.voidptr
                mv = sip.voidptr(self._siparray, self._capa*self._nfields*8)
            # note that we perform the slicing by using only _size rows
            nd = np.frombuffer(mv, dtype=np.float64, count=self._size*self._nfields)
            return nd.reshape((-1, self._nfields))
        else:
            return self._ndarray[:self._size]

    def instances(self):
        # this returns an iterable container of Klass instances.
        # for "use_ptr_to_array" mode, such a container may not
        # be required at all, so its creation is deferred
        if self.use_sip_array:
            if self._size == self._capa:
                # avoiding slicing when it's not necessary
                # handles the case where sip.array had a buggy
                # slicing implementation 
                return self._siparray
            else:
                # this is a view
                return self._siparray[:self._size]

        if self._objs is None:
            self._objs = self._wrap_instances(self._ndarray)

        if self._size == self._capa:
            return self._objs
        else:
            # this is a shallow copy
            return self._objs[:self._size]

    def drawargs(self):
        # returns arguments to apply to the respective drawPrimitives() functions
        if self.use_ptr_to_array:
            if self._capa > 0:
                # wrap memory only if it is safe to do so
                ptr = compat.wrapinstance(self._ndarray.ctypes.data, self._Klass)
            else:
                # shiboken translates None <--> nullptr
                # alternatively, we could instantiate a dummy _Klass()
                ptr = None
            return ptr, self._size

        else:
            return self.instances(),
