from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from ..Qt import QtWidgets
import typing

__all__ = ['MatplotlibWidget']

class MatplotlibWidget(QtWidgets.QWidget):
    
    @typing.overload
    def __init__(self, figsize=(5.0, 4.0), dpi=100, parent=None, flags=None):
        pass

    @typing.overload
    def __init__(self, parent=None, flags=None):
        pass

    def __init__(self, *args, **kwargs):
        if (
            (args and not isinstance(args[0], QtWidgets.QWidget))  # figsize is provided in the positional args
            or
            (kwargs and ("figsize" in kwargs or "dpi" in kwargs))  # figsize or dpi are provided in keyword args
        ):  # If figsize or dpi are provided
            figsize = args[0] if len(args) > 0 else kwargs.get("figsize", (5.0, 4.0))
            dpi = args[1] if len(args) > 1 else kwargs.get("dpi", 100)
            parent = args[2] if len(args) > 2 else kwargs.get("parent", None)
            flags = args[3] if len(args) > 3 else kwargs.get("flags", None)
        else:
            figsize = (5.0, 4.0)
            dpi = 100
            parent = args[0] if len(args) > 0 else kwargs.get("parent", None)
            flags = args[1] if len(args) > 1 else kwargs.get("flags", None)
        super().__init__(parent, flags)
        self.figsize = figsize
        self.dpi = dpi

    def getFigure(self):
        return self.fig
        
    def draw(self):
        self.canvas.draw()
