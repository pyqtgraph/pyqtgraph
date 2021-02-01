import os.path as op

from ..Qt import QtGui

__all__ = ['getGraphIcon', 'getGraphPixmap']

_ICON_REGISTRY = {}


class GraphIcon:
    """An icon place holder for lazy loading of QIcons"""

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


def getPixmap(name, size=(20, 20)):
    """Historic `getPixmap` function

    (eg. getPixmap('auto') loads pyqtgraph/icons/auto.png)
    """
    return getGraphPixmap(name, size=size)


# Note: List all graph icons here ...
auto = GraphIcon("auto.png")
ctrl = GraphIcon("ctrl.png")
default = GraphIcon("default.png")
invisibleEye = GraphIcon("invisibleEye.svg")
lock = GraphIcon("lock.png")
