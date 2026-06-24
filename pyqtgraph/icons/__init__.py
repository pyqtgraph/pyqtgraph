import os.path as op
import warnings

from ..Qt import QtGui, QtWidgets

__all__ = ['getGraphIcon', 'getGraphPixmap']

_ICON_REGISTRY = {}


class GraphIcon:
    """An icon place holder for lazy loading of QIcons

    The icon must reside in the icons folder and the path refers to the full
    name including suffix of the icon file, e.g.:

        tiny = GraphIcon("tiny.png")

    Icons can be later retrieved via the function `getGraphIcon` and providing
    the name:

        tiny = getGraphIcon("tiny")
    """

    def __init__(self, path):
        self._path = path
        name = path.split('.')[0]
        _ICON_REGISTRY[name] = self
        self._icon = None

    def _build_qicon(self):
        icon = QtGui.QIcon(op.join(op.dirname(__file__), self._path))
        name = self._path.split('.')[0]
        _ICON_REGISTRY[name] = icon
        self._icon = icon

    @property
    def qicon(self):
        if self._icon is None:
            self._build_qicon()

        return self._icon


def getGraphIcon(name):
    """Return a `PyQtGraph` icon from the registry by `name`"""
    icon = _ICON_REGISTRY[name]
    if isinstance(icon, GraphIcon):
        icon = icon.qicon
        _ICON_REGISTRY[name] = icon

    return icon


def getGraphPixmap(name, size=(20, 20)):
    """Return a `QPixmap` from the registry by `name`"""
    icon = getGraphIcon(name)

    return icon.pixmap(*size)


# Note: List all graph icons here ...
# GraphIcon registers itself in _ICON_REGISTRY as a side effect of __init__,
# so these are intentionally not bound to a name (avoids CodeQL "unused
# global variable" warnings); 
# Look them up via getGraphIcon(name) or getGraphPixmap
auto = GraphIcon("auto.png")
ctrl = GraphIcon("ctrl.png")
default = GraphIcon("default.png")
GraphIcon("delete.png")
invisibleEye = GraphIcon("invisibleEye.svg")
lock = GraphIcon("lock.png")
GraphIcon("rename.png")
GraphIcon("revert_default.png")
GraphIcon("set_default.png")
GraphIcon("unlock.png")
GraphIcon("visibleEye.svg")
