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

    parent_default = None
    figsize_default = (5.0, 4.0)
    dpi_default = 100

    @typing.overload
    def __init__(self, figsize=(5.0, 4.0), dpi=100, parent=None):
        pass

    @typing.overload
    def __init__(self, parent=None, figsize=(5.0, 4.0), dpi=100):
        pass

    def __init__(self, *args, **kwargs):
        if (args and not isinstance(args[0], QtWidgets.QWidget)):
            figsize = args[0] if len(args) > 0 \
                else kwargs.get("figsize", MatplotlibWidget.figsize_default)
            dpi = args[1] if len(args) > 1 \
                else kwargs.get("dpi", MatplotlibWidget.dpi_default)
            parent = args[2] if len(args) > 2 \
                else kwargs.get("parent", MatplotlibWidget.parent_default)
        else:
            parent = args[0] if len(args) > 0 \
                else kwargs.get("parent", MatplotlibWidget.parent_default)
            figsize = args[1] if len(args) > 1 \
                else kwargs.get("figsize", MatplotlibWidget.figsize_default)
            dpi = args[2] if len(args) > 2 \
                else kwargs.get("dpi", MatplotlibWidget.dpi_default)
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
