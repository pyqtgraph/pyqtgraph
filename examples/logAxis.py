# -*- coding: utf-8 -*-
"""
Test programmatically setting log transformation modes.
"""
import initExample ## Add path to library (just for examples; you do not need this)

import numpy as np
import pyqtgraph as pg

app = pg.mkQApp("Log Axis Example")

w = pg.GraphicsLayoutWidget(show=True)
w.resize(800,800)
w.setWindowTitle('pyqtgraph example: Log Axis, or How to Recognise Different Types of Curves from Quite a Long Way Away')

p0 = w.addPlot(0,0, title="Linear")
p1 = w.addPlot(0,1, title="X Semilog")
p2 = w.addPlot(1,0, title="Y Semilog")
p3 = w.addPlot(1,1, title="XY Log")
p1.setLogMode(True, False)
p2.setLogMode(False, True)
p3.setLogMode(True, True)

x = np.logspace(-1, 1, 1000) # 1000 points from 0.01 to 100
plotdata = (
    ('1 / 3x'    , '#ff9d47', 1./(3*x) ),
    ('sqrt x'    , '#b3cf00', 1/np.sqrt(x) ),
    ('exp. decay', '#00a0b5', 5 * np.exp(-x/1) ), 
    ('-log x'    , '#a54dff', - np.log10(x) )
)
p0.addLegend( offset=(-20,20) )
for p in (p0, p1, p2, p3):
    p.showGrid(True, True)
    p.showAxes(True, size=(40,None))
    for name, color, y in plotdata:
        pen = pg.mkPen(color, width=2)
        p.plot( x,y, pen=pen, name=name )

w.show()

if __name__ == '__main__':
    pg.exec()
