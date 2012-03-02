# -*- coding: utf-8 -*-
import sys, os

## Make sure pyqtgraph is importable
p = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(p, '..', '..')
sys.path.insert(0, p)


from pyqtgraph.flowchart import Flowchart
from pyqtgraph.Qt import QtGui

#import pyqtgraph.flowchart as f

app = QtGui.QApplication([])

#TETRACYCLINE = True

fc = Flowchart(terminals={
    'dataIn': {'io': 'in'},
    'dataOut': {'io': 'out'}    
})
w = fc.widget()
w.resize(400,200)
w.show()

n1 = fc.createNode('Add')
n2 = fc.createNode('Subtract')
n3 = fc.createNode('Abs')
n4 = fc.createNode('Add')

fc.connectTerminals(fc.dataIn, n1.A)
fc.connectTerminals(fc.dataIn, n1.B)
fc.connectTerminals(fc.dataIn, n2.A)
fc.connectTerminals(n1.Out, n4.A)
fc.connectTerminals(n1.Out, n2.B)
fc.connectTerminals(n2.Out, n3.In)
fc.connectTerminals(n3.Out, n4.B)
fc.connectTerminals(n4.Out, fc.dataOut)


def process(**kargs):
    return fc.process(**kargs)

    
print process(dataIn=7)

fc.setInput(dataIn=3)

s = fc.saveState()
fc.clear()

fc.restoreState(s)

fc.setInput(dataIn=3)

#f.NodeMod.TETRACYCLINE = False

if sys.flags.interactive == 0:
    app.exec_()

