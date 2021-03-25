from warnings import warn

from .. import getConfigOption

def getNumbaFunctions():
    if getConfigOption("useNumba"):
        try:
            import numba
        except ImportError:
            warn("numba library could not be loaded, but 'useNumba' is set.")
            return None

        if numba.version_info.full < (0, 53):
            warn(f"pyqtgraph requires numba version >= 0.53  (your version is {numba.version_info.string})")
            return None

        from .. import functions_numba
        return functions_numba
    else:
        return None
