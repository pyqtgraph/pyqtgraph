# -*- coding: utf-8 -*-
import initExample ## Add path to library (just for examples; you do not need this)


from pyqtgraph.flowchart import Flowchart
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np

app = QtGui.QApplication([])


win = QtGui.QMainWindow()
cw = QtGui.QWidget()
win.setCentralWidget(cw)
layout = QtGui.QGridLayout()
cw.setLayout(layout)

fc = Flowchart(terminals={
    'dataIn': {'io': 'in'},
    'dataOut': {'io': 'out'}    
})
w = fc.widget()

layout.addWidget(fc.widget(), 0, 0, 2, 1)

pw1 = pg.PlotWidget()
pw2 = pg.PlotWidget()
layout.addWidget(pw1, 0, 1)
layout.addWidget(pw2, 1, 1)

win.show()


data = np.random.normal(size=1000)
data[200:300] += 1
data += np.sin(np.linspace(0, 100, 1000))

fc.setInput(dataIn=data)

pw1Node = fc.createNode('PlotWidget', pos=(0, -150))
pw1Node.setPlot(pw1)

pw2Node = fc.createNode('PlotWidget', pos=(150, -150))
pw2Node.setPlot(pw2)

fNode = fc.createNode('GaussianFilter', pos=(0, 0))
fc.connectTerminals(fc.dataIn, fNode.In)
fc.connectTerminals(fc.dataIn, pw1Node.In)
fc.connectTerminals(fNode.Out, pw2Node.In)
fc.connectTerminals(fNode.Out, fc.dataOut)


#n1 = fc.createNode('Add', pos=(0,-80))
#n2 = fc.createNode('Subtract', pos=(140,-10))
#n3 = fc.createNode('Abs', pos=(0, 80))
#n4 = fc.createNode('Add', pos=(140,100))

#fc.connectTerminals(fc.dataIn, n1.A)
#fc.connectTerminals(fc.dataIn, n1.B)
#fc.connectTerminals(fc.dataIn, n2.A)
#fc.connectTerminals(n1.Out, n4.A)
#fc.connectTerminals(n1.Out, n2.B)
#fc.connectTerminals(n2.Out, n3.In)
#fc.connectTerminals(n3.Out, n4.B)
#fc.connectTerminals(n4.Out, fc.dataOut)


#def process(**kargs):
    #return fc.process(**kargs)

    
#print process(dataIn=7)

#fc.setInput(dataIn=3)

#s = fc.saveState()
#fc.clear()

#fc.restoreState(s)

#fc.setInput(dataIn=3)


## Start Qt event loop unless running in interactive mode or using pyside.
import sys
if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    app.exec_()
