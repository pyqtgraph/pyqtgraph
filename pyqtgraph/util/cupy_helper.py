import os
from warnings import warn

from .. import getConfigOption


def getCupy():
    if getConfigOption("useCupy"):
        try:
            import cupy
        except ImportError:
            warn("cupy library could not be loaded, but 'useCupy' is set.")
            return None
        if os.name == "nt" and cupy.cuda.runtime.runtimeGetVersion() < 11000:
            warn("In Windows, CUDA toolkit should be version 11 or higher, or some functions may misbehave.")
        return cupy
    else:
        return None
