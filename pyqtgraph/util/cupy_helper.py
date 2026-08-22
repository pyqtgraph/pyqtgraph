import os
from warnings import warn
from types import ModuleType

from .. import getConfigOption


def getCupy() -> ModuleType | None:
    if getConfigOption("useCupy"):
        try:
            import cupy  # pyright: ignore[reportMissingImports]
        except ImportError:
            warn("cupy library could not be loaded, but 'useCupy' is set.")
            return None
        else:
            if os.name == "nt" and cupy.cuda.runtime.runtimeGetVersion() < 11000:  # pyright: ignore[reportUnknownMemberType]
                warn("In Windows, CUDA toolkit should be version 11 or higher, or some functions may misbehave.")
        return cupy
    else:
        return None
