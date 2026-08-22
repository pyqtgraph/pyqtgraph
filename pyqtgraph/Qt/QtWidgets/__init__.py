import importlib
from .. import QT_LIB
module = importlib.import_module(f'{QT_LIB}.QtWidgets')

def __getattr__(name):
    x = getattr(module, name)
    globals()[name] = x
    return x
