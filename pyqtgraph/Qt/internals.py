import ctypes
import itertools
import sys

import numpy as np

from . import QT_LIB, QtCore, QtGui, compat

__all__ = ["get_qpainterpath_element_array"]

if QT_LIB.startswith('PyQt'):
    from . import sip
    qt_version_info = tuple((QtCore.QT_VERSION >> i) & 0xff for i in [16,8,0])
elif QT_LIB == 'PySide2':
    from PySide2 import __version_info__ as pyside_version_info
    qt_version_info = QtCore.__version_info__
elif QT_LIB == 'PySide6':
    from PySide6 import __version_info__ as pyside_version_info
    qt_version_info = QtCore.__version_info__


class Element(ctypes.Structure):
    _fields_=  [('x', ctypes.c_double), ('y', ctypes.c_double), ('c', ctypes.c_int)]

class QArrayData(ctypes.Structure):
    pass

class QPainterPathPrivate(ctypes.Structure):
    pass

if qt_version_info[0] == 5:
    QArrayData._fields_ = [
        ("ref", ctypes.c_int),
        ("size", ctypes.c_int),
        ("alloc", ctypes.c_uint, 31),
        ("offset", ctypes.c_ssize_t),
    ]

    QPainterPathPrivate._fields_ = [
        ("ref", ctypes.c_int),
        ("adata", ctypes.POINTER(QArrayData)),
    ]

elif qt_version_info[0] == 6:
    QArrayData._fields_ = [
        ("ref", ctypes.c_int),
        ("flags", ctypes.c_uint),
        ("alloc", ctypes.c_ssize_t),
    ]

    QPainterPathPrivate._fields_ = [
        ("ref", ctypes.c_int),
        ("adata", ctypes.POINTER(QArrayData)),
        ("data", ctypes.c_void_p),
        ("size", ctypes.c_ssize_t),
    ][int(qt_version_info >= (6, 10)):]

def get_qpainterpath_element_array(qpath, nelems=None):
    resize = nelems is not None
    if resize:
        qpath.reserve(nelems)

    ptr = ctypes.c_void_p.from_address(compat.unwrapinstance(qpath))
    if not ptr:
        return np.zeros(0, dtype=Element)

    ppp = ctypes.cast(ptr, ctypes.POINTER(QPainterPathPrivate)).contents

    if qt_version_info[0] == 5:
        qad = ppp.adata.contents
        eptr = ctypes.addressof(qad) + qad.offset
        if resize:
            qad.size = nelems
    elif qt_version_info[0] == 6:
        eptr = ppp.data
        if resize:
            ppp.size = nelems
    else:
        raise NotImplementedError

    nelems = qpath.elementCount()
    buf = (Element * nelems).from_address(eptr)
    return np.frombuffer(buf, dtype=Element)

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


_qbytearray_leaks = None

def qbytearray_leaks() -> bool:
    global _qbytearray_leaks

    if _qbytearray_leaks is None:
        # When PySide{2,6} is built without Py_LIMITED_API,
        # it leaks memory when a memory view to a QByteArray
        # object is taken.
        # See https://github.com/pyqtgraph/pyqtgraph/issues/3265
        # and PYSIDE-3031
        # Note: official builds of PySide{2,6} by Qt are built with
        # the limited api, and thus do not leak.
        if QT_LIB.startswith("PySide"):
            # probe whether QByteArray leaks
            qba = QtCore.QByteArray()
            ref0 = sys.getrefcount(qba)
            memoryview(qba)
            _qbytearray_leaks = sys.getrefcount(qba) > ref0
        else:
            _qbytearray_leaks = False

    return _qbytearray_leaks
