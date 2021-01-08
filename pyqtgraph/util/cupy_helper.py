from pyqtgraph import getConfigOption


def getCupy():
    if getConfigOption("useCupy"):
        import cupy
        return cupy
    else:
        return None
