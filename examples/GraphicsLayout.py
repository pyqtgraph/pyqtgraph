## Add path to library (just for examples; you do not need this)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import user

app = QtGui.QApplication([])
view = pg.GraphicsView()
l = pg.GraphicsLayout(border=pg.mkPen(0, 0, 255))
view.setCentralItem(l)
view.show()

## Add 3 plots into the first row (automatic position)
p1 = l.addPlot()
p2 = l.addPlot()
p3 = l.addPlot()

## Add a viewbox into the second row (automatic position)
l.nextRow()
vb = l.addViewBox(colspan=3)

## Add 2 more plots into the third row (manual position)
p4 = l.addPlot(row=2, col=0)
p5 = l.addPlot(row=2, col=1, colspan=2)



## show some content
p1.plot([1,3,2,4,3,5])
p2.plot([1,3,2,4,3,5])
p3.plot([1,3,2,4,3,5])
p4.plot([1,3,2,4,3,5])
p5.plot([1,3,2,4,3,5])

b = QtGui.QGraphicsRectItem(0, 0, 1, 1)
b.setPen(pg.mkPen(255,255,0))
vb.addItem(b)
vb.setRange(QtCore.QRectF(-1, -1, 3, 3))



## Start Qt event loop unless running in interactive mode.
if sys.flags.interactive != 1:
    app.exec_()
