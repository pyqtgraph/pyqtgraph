"""
This example demonstrates the use of RemoteGraphicsView to improve performance in
applications with heavy load. It works by starting a second process to handle 
all graphics rendering, thus freeing up the main process to do its work.

In this example, the update() function is very expensive and is called frequently.
After update() generates a new set of data, it can either plot directly to a local
plot (bottom) or remotely via a RemoteGraphicsView (top), allowing speed comparison
between the two cases. IF you have a multi-core CPU, it should be obvious that the 
remote case is much faster.
"""

import argparse
import itertools

import numpy as np
from utils import FrameCounter

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', default=float('inf'), type=float,
    help="Number of iterations to run before exiting"
)
args = parser.parse_args()
iterations_counter = itertools.count()

app = pg.mkQApp()

view = pg.widgets.RemoteGraphicsView.RemoteGraphicsView()
pg.setConfigOptions(antialias=True)  ## this will be expensive for the local plot
view.pg.setConfigOptions(antialias=True)  ## prettier plots at no cost to the main process! 
view.setWindowTitle('pyqtgraph example: RemoteSpeedTest')

app.aboutToQuit.connect(view.close)

label = QtWidgets.QLabel()
rcheck = QtWidgets.QCheckBox('plot remote')
rcheck.setChecked(True)
lcheck = QtWidgets.QCheckBox('plot local')
lplt = pg.PlotWidget()
layout = pg.LayoutWidget()
layout.addWidget(rcheck)
layout.addWidget(lcheck)
layout.addWidget(label)
layout.addWidget(view, row=1, col=0, colspan=3)
layout.addWidget(lplt, row=2, col=0, colspan=3)
layout.resize(800,800)
layout.show()

## Create a PlotItem in the remote process that will be displayed locally
rplt = view.pg.PlotItem()
rplt._setProxyOptions(deferGetattr=True)  ## speeds up access to rplt.plot
view.setCentralItem(rplt)

def update():
    if next(iterations_counter) > args.iterations:
        timer.stop()
        app.quit()
        return None

    data = np.random.normal(size=(10000,50)).sum(axis=1)
    data += 5 * np.sin(np.linspace(0, 10, data.shape[0]))
    
    if rcheck.isChecked():
        rplt.plot(data, clear=True, _callSync='off')  ## We do not expect a return value.
                                                      ## By turning off callSync, we tell
                                                      ## the proxy that it does not need to 
                                                      ## wait for a reply from the remote
                                                      ## process.
    if lcheck.isChecked():
        lplt.plot(data, clear=True)

    framecnt.update()
        
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

framecnt = FrameCounter()
framecnt.sigFpsUpdate.connect(lambda fps : label.setText(f"Generating {fps:.1f}"))

if __name__ == '__main__':
    pg.exec()
