import os.path as op

from ..Qt import QtGui

__all__ = ['get_graph_icon']

_ICON_REGISTRY = {}


class GraphIcon:
    """An icon place holder for lazy loading of QIcons"""

    def __init__(self, path):
        self._path = path
        self._icon = None
        name = path.split('.')[0]
        _ICON_REGISTRY[name] = self

    @property
    def qicon(self):
        if self._icon is None:
            self._icon = QtGui.QIcon(op.join(op.dirname(__file__), self._path))

        return self._icon


def get_graph_icon(name):
    """Return a `PyQtGraph icon from the registry by `name`"""
    icon = _ICON_REGISTRY[name]
    if isinstance(icon, GraphIcon):
        icon = icon.qicon
        _ICON_REGISTRY[name] = icon

    return icon

# Note: List all graph icons here ...
invisibleEye = GraphIcon("invisibleEye.svg")
