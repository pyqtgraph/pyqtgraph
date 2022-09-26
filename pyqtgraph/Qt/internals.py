import ctypes
import numpy as np
from . import QT_LIB
from . import compat

__all__ = ["get_qpainterpath_element_array"]

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
