from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from ..Qt import QtWidgets
import typing

__all__ = ['MatplotlibWidget']

class MatplotlibWidget(QtWidgets.QWidget):
    """
    Implements a Matplotlib figure inside a QWidget.
    Use getFigure() and redraw() to interact with matplotlib.

    Example::
    
        mw = MatplotlibWidget()
        subplot = mw.getFigure().add_subplot(111)
        subplot.plot(x,y)
        mw.draw()
    """
    
    @typing.overload
    def __init__(self, figsize=(5.0, 4.0), dpi=100, parent=None):
        pass

    @typing.overload
    def __init__(self, parent=None):
        pass

    def __init__(self, *args, **kwargs):
        if (
            (args and not isinstance(args[0], QtWidgets.QWidget))
            or
            (kwargs and "figsize" in kwargs or "dpi" in kwargs)
        ):
            figsize = args[0] if len(args) > 0 else kwargs.get("figsize", (5.0, 4.0))
            dpi = args[1] if len(args) > 1 else kwargs.get("dpi", 100)
            parent = args[2] if len(args) > 2 else kwargs.get("parent", None)
        else:
            parent = args[0] if len(args) > 0 else kwargs.get("parent", None)
            figsize = kwargs.get("figsize", (5.0, 4.0))
            dpi = kwargs.get("dpi", 100)
        super().__init__(parent)

        self.fig = Figure(figsize, dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.vbox = QtWidgets.QVBoxLayout()
        self.vbox.addWidget(self.toolbar)
        self.vbox.addWidget(self.canvas)

        self.setLayout(self.vbox)

    def getFigure(self):
        return self.fig
        
    def draw(self):
        self.canvas.draw()
