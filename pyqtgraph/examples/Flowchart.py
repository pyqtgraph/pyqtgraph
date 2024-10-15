"""
This example demonstrates a very basic use of flowcharts: filter data,
displaying both the input and output of the filter. The behavior of
the filter can be reprogrammed by the user.

Basic steps are:
  - create a flowchart and two plots
  - input noisy data to the flowchart
  - flowchart connects data to the first plot, where it is displayed
  - add a gaussian filter to lowpass the data, then display it in the second plot.
"""

import numpy as np

import pyqtgraph as pg
from pyqtgraph.flowchart import Flowchart
from pyqtgraph.Qt import QtWidgets

app = pg.mkQApp("Flowchart Example")

## Create main window with grid layout
win = QtWidgets.QMainWindow()
win.setWindowTitle('pyqtgraph example: Flowchart')
cw = QtWidgets.QWidget()
win.setCentralWidget(cw)
layout = QtWidgets.QGridLayout()
cw.setLayout(layout)

## Create flowchart, define input/output terminals
fc = Flowchart(terminals={
    'dataIn': {'io': 'in'},
    'dataOut': {'io': 'out'}    
})
w = fc.widget()

## Add flowchart control panel to the main window
layout.addWidget(fc.widget(), 0, 0, 2, 1)

## Add two plot widgets
pw1 = pg.PlotWidget()
pw2 = pg.PlotWidget()
layout.addWidget(pw1, 0, 1)
layout.addWidget(pw2, 1, 1)

win.show()

## generate signal data to pass through the flowchart
data = np.random.normal(size=1000)
data[200:300] += 1
data += np.sin(np.linspace(0, 100, 1000))

## Feed data into the input terminal of the flowchart
fc.setInput(dataIn=data)

## populate the flowchart with a basic set of processing nodes. 
## (usually we let the user do this)
plotList = {'Top Plot': pw1, 'Bottom Plot': pw2}

pw1Node = fc.createNode('PlotWidget', pos=(0, -150))
pw1Node.setPlotList(plotList)
pw1Node.setPlot(pw1)

pw2Node = fc.createNode('PlotWidget', pos=(150, -150))
pw2Node.setPlotList(plotList)
pw2Node.setPlot(pw2)

fNode = fc.createNode('GaussianFilter', pos=(0, 0))
fNode.ctrls['sigma'].setValue(5)
fc.connectTerminals(fc['dataIn'], fNode['In'])
fc.connectTerminals(fc['dataIn'], pw1Node['In'])
fc.connectTerminals(fNode['Out'], pw2Node['In'])
fc.connectTerminals(fNode['Out'], fc['dataOut'])

if __name__ == '__main__':
    pg.exec()
