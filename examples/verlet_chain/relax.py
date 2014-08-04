import ctypes
import os

so = os.path.join(os.path.dirname(__file__), 'maths.so')
lib = ctypes.CDLL(so)

lib.relax.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ]

def relax(pos, links, mrel1, mrel2, lengths, push, pull, iters):
    nlinks = links.shape[0]
    lib.relax(pos.ctypes, links.ctypes, mrel1.ctypes, mrel2.ctypes, lengths.ctypes, push.ctypes, pull.ctypes, nlinks, iters)
    

