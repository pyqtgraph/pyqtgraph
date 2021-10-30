"""
Demonstrate programmatic setting of log transformation modes.
"""

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
# configure logarithmic axis scaling:
p1.setLogMode(True, False)
p2.setLogMode(False, True)
p3.setLogMode(True, True)

# 1000 points from 0.1 to 10, chosen to give a compatible range of values across curves:
x = np.logspace(-1, 1, 1000) 
plotdata = ( # legend entry, color, and plotted equation:
    ('1 / 3x'    , '#ff9d47', 1./(3*x) ),
    ('sqrt x'    , '#b3cf00', 1/np.sqrt(x) ),
    ('exp. decay', '#00a0b5', 5 * np.exp(-x/1) ), 
    ('-log x'    , '#a54dff', - np.log10(x) )
)
p0.addLegend(offset=(-20,20)) # include legend only in top left plot
for p in (p0, p1, p2, p3):    # draw identical numerical data in all four plots
    p.showGrid(True, True)    # turn on grid for all four plots
    p.showAxes(True, size=(40,None)) # show a full frame, and reserve identical room for y labels
    for name, color, y in plotdata:  # draw all four curves as defined in plotdata
        pen = pg.mkPen(color, width=2)
        p.plot( x,y, pen=pen, name=name )

w.show()

if __name__ == '__main__':
    pg.exec()
