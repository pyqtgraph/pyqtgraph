"""
Deprecated: this module has been renamed to
:mod:`pyqtgraph.graphicsItems.GroupItem`, and ``ItemGroup`` itself has been
renamed to ``GroupItem``. Both names are kept here as aliases for backward
compatibility and will be removed in a future version of pyqtgraph.
"""
from .._deprecated_names import RENAMED_SYMBOLS
from .._deprecation import redirect_module

redirect_module(globals(), __name__, RENAMED_SYMBOLS)
