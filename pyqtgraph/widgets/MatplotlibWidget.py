from ..Qt import QtGui, QtCore
from .. import Qt
import matplotlib

if Qt.QT_LIB == Qt.LIB_PYSIDE: 
    matplotlib.rcParams['backend.qt4']='PySide'
elif Qt.QT_LIB == Qt.LIB_PYQT4:
    #not sure whether to choose 'PyQt4' or 'PyQt4v2' here. However, previous
    #versions of this code only dealt with PySide, so *presumably* the default
    #is to do the right thing for PyQt4...
    pass
elif Qt.QT_LIB == Qt.LIB_PYQT5:
    matplotlib.rcParams['backend.qt5']='PyQt5'

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure

class MatplotlibWidget(QtGui.QWidget):
    """
    Implements a Matplotlib figure inside a QWidget.
    Use getFigure() and redraw() to interact with matplotlib.
    
    Example::
    
        mw = MatplotlibWidget()
        subplot = mw.getFigure().add_subplot(111)
        subplot.plot(x,y)
        mw.draw()
    """
    
    def __init__(self, size=(5.0, 4.0), dpi=100):
        QtGui.QWidget.__init__(self)
        self.fig = Figure(size, dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addWidget(self.toolbar)
        self.vbox.addWidget(self.canvas)
        
        self.setLayout(self.vbox)

    def getFigure(self):
        return self.fig
        
    def draw(self):
        self.canvas.draw()
