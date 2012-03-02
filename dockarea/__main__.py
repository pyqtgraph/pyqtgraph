import sys

## Make sure pyqtgraph is importable
p = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(p, '..', '..')
sys.path.insert(0, p)

from pyqtgraph.Qt import QtCore, QtGui

from DockArea import *
from Dock import *

app = QtGui.QApplication([])
win = QtGui.QMainWindow()
area = DockArea()
win.setCentralWidget(area)
win.resize(800,800)
from Dock import Dock
d1 = Dock("Dock1", size=(200,200))
d2 = Dock("Dock2", size=(100,100))
d3 = Dock("Dock3", size=(1,1))
d4 = Dock("Dock4", size=(50,50))
d5 = Dock("Dock5", size=(100,100))
d6 = Dock("Dock6", size=(300,300))
area.addDock(d1, 'left')
area.addDock(d2, 'right')
area.addDock(d3, 'bottom')
area.addDock(d4, 'right')
area.addDock(d5, 'left', d1)
area.addDock(d6, 'top', d4)

area.moveDock(d6, 'above', d4)
d3.hideTitleBar()

print "===build complete===="

for d in [d1, d2, d3, d4, d5]:
    w = QtGui.QWidget()
    l = QtGui.QVBoxLayout()
    w.setLayout(l)
    btns = []
    for i in range(4):
        btns.append(QtGui.QPushButton("%s Button %d"%(d.name(), i)))
        l.addWidget(btns[-1])
    d.w = (w, l, btns)
    d.addWidget(w)



import pyqtgraph as pg
p = pg.PlotWidget()
d6.addWidget(p)

print "===widgets added==="


#s = area.saveState()


#print "\n\n-------restore----------\n\n"
#area.restoreState(s)
s = None
def save():
    global s
    s = area.saveState()
    
def load():
    global s
    area.restoreState(s)


#d6.container().setCurrentIndex(0)
#d2.label.setTabPos(40)

#win2 = QtGui.QMainWindow()
#area2 = DockArea()
#win2.setCentralWidget(area2)
#win2.resize(800,800)


win.show()
#win2.show()

