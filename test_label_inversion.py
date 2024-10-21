import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

app = QtGui.QApplication([])

# Create a plot widget
plot = pg.PlotWidget()
plot.show()

# Add a label item
label = pg.LabelItem(text='Test Label', color='red')
plot.addItem(label)

app.exec_()
