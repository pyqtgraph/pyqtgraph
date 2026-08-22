import importlib
from .. import QtVersionInfo, QT_LIB
if QtVersionInfo[0] >= 6:
    module = importlib.import_module(f'{QT_LIB}.QtOpenGL')
else:
    module = importlib.import_module(f'{QT_LIB}.QtGui')

def __getattr__(name):
    x = getattr(module, name)
    globals()[name] = x
    return x
