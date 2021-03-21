# -*- coding: utf-8 -*-
"""
This example demonstrates the SpinBox widget, which is an extension of 
QDoubleSpinBox providing some advanced features:

  * SI-prefixed units
  * Non-linear stepping modes
  * Bounded/unbounded values

"""
import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import ast

app = pg.mkQApp("SpinBox Example")


spins = [
    ("Floating-point spin box, min=0, no maximum.<br>Non-finite values (nan, inf) are permitted.",
     pg.SpinBox(value=5.0, bounds=[0, None], finite=False)),
    ("Integer spin box, dec stepping<br>(1-9, 10-90, 100-900, etc), decimals=4", 
     pg.SpinBox(value=10, int=True, dec=True, minStep=1, step=1, decimals=4)),
    ("Float with SI-prefixed units<br>(n, u, m, k, M, etc)", 
     pg.SpinBox(value=0.9, suffix='V', siPrefix=True)),
    ("Float with SI-prefixed units,<br>dec step=0.1, minStep=0.1", 
     pg.SpinBox(value=1.0, suffix='PSI', siPrefix=True, dec=True, step=0.1, minStep=0.1)),
    ("Float with SI-prefixed units,<br>dec step=0.5, minStep=0.01", 
     pg.SpinBox(value=1.0, suffix='V', siPrefix=True, dec=True, step=0.5, minStep=0.01)),
    ("Float with SI-prefixed units,<br>dec step=1.0, minStep=0.001", 
     pg.SpinBox(value=1.0, suffix='V', siPrefix=True, dec=True, step=1.0, minStep=0.001)),
    ("Float with SI prefix but no suffix",
     pg.SpinBox(value=1e9, siPrefix=True)),
    ("Float with custom formatting", 
     pg.SpinBox(value=23.07, format='${value:0.02f}',
                regex='\$?(?P<number>(-?\d+(\.\d+)?)|(-?\.\d+))$')),
    ("Int with suffix",
     pg.SpinBox(value=999, step=1, int=True, suffix="V")),
    ("Int with custom formatting", 
     pg.SpinBox(value=4567, step=1, int=True, bounds=[0,None], format='0x{value:X}', 
                regex='(0x)?(?P<number>[0-9a-fA-F]+)$',
                evalFunc=lambda s: ast.literal_eval('0x'+s))),
    ("Integer with bounds=[10, 20] and wrapping",
     pg.SpinBox(value=10, bounds=[10, 20], int=True, minStep=1, step=1, wrapping=True)),
]


win = QtGui.QMainWindow()
win.setWindowTitle('pyqtgraph example: SpinBox')
cw = QtGui.QWidget()
layout = QtGui.QGridLayout()
cw.setLayout(layout)
win.setCentralWidget(cw)
win.show()
#win.resize(300, 600)
changingLabel = QtGui.QLabel()  ## updated immediately
changedLabel = QtGui.QLabel()   ## updated only when editing is finished or mouse wheel has stopped for 0.3sec
changingLabel.setMinimumWidth(200)
font = changingLabel.font()
font.setBold(True)
font.setPointSize(14)
changingLabel.setFont(font)
changedLabel.setFont(font)
labels = []


def valueChanged(sb):
    changedLabel.setText("Final value: %s" % str(sb.value()))

def valueChanging(sb, value):
    changingLabel.setText("Value changing: %s" % str(sb.value()))

    
for text, spin in spins:
    label = QtGui.QLabel(text)
    labels.append(label)
    layout.addWidget(label)
    layout.addWidget(spin)
    spin.sigValueChanged.connect(valueChanged)
    spin.sigValueChanging.connect(valueChanging)

layout.addWidget(changingLabel, 0, 1)
layout.addWidget(changedLabel, 2, 1)


#def mkWin():
    #win = QtGui.QMainWindow()
    #g = QtGui.QFormLayout()
    #w = QtGui.QWidget()
    #w.setLayout(g)
    #win.setCentralWidget(w)
    #s1 = SpinBox(value=5, step=0.1, bounds=[-1.5, None], suffix='units')
    #t1 = QtGui.QLineEdit()
    #g.addRow(s1, t1)
    #s2 = SpinBox(value=10e-6, dec=True, step=0.1, minStep=1e-6, suffix='A', siPrefix=True)
    #t2 = QtGui.QLineEdit()
    #g.addRow(s2, t2)
    #s3 = SpinBox(value=1000, dec=True, step=0.5, minStep=1e-6, bounds=[1, 1e9], suffix='Hz', siPrefix=True)
    #t3 = QtGui.QLineEdit()
    #g.addRow(s3, t3)
    #s4 = SpinBox(int=True, dec=True, step=1, minStep=1, bounds=[-10, 1000])
    #t4 = QtGui.QLineEdit()
    #g.addRow(s4, t4)

    #win.show()

    #import sys
    #for sb in [s1, s2, s3,s4]:

        ##QtCore.QObject.connect(sb, QtCore.SIGNAL('valueChanged(double)'), lambda v: sys.stdout.write(str(sb) + " valueChanged\n"))
        ##QtCore.QObject.connect(sb, QtCore.SIGNAL('editingFinished()'), lambda: sys.stdout.write(str(sb) + " editingFinished\n"))
        #sb.sigValueChanged.connect(valueChanged)
        #sb.sigValueChanging.connect(valueChanging)
        #sb.editingFinished.connect(lambda: sys.stdout.write(str(sb) + " editingFinished\n"))
    #return win, w, [s1, s2, s3, s4]
#a = mkWin()


#def test(n=100):
    #for i in range(n):
        #win, w, sb = mkWin()
        #for s in sb:
            #w.setParent(None)
            #s.setParent(None)
            #s.valueChanged.disconnect()
            #s.editingFinished.disconnect()


if __name__ == '__main__':
    pg.Qt.QtWidgets.QApplication.exec_()
