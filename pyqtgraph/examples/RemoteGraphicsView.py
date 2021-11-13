"""
Very simple example demonstrating RemoteGraphicsView.

This allows graphics to be rendered in a child process and displayed in the 
parent, which can improve CPU usage on multi-core processors.
"""

import pyqtgraph as pg
from pyqtgraph.widgets.RemoteGraphicsView import RemoteGraphicsView

app = pg.mkQApp()

## Create the widget
v = RemoteGraphicsView(debug=False)  # setting debug=True causes both processes to print information
                                    # about interprocess communication
v.show()
v.setWindowTitle('pyqtgraph example: RemoteGraphicsView')

## v.pg is a proxy to the remote process' pyqtgraph module. All attribute 
## requests and function calls made with this object are forwarded to the
## remote process and executed there. See pyqtgraph.multiprocess.remoteproxy
## for more inormation.
plt = v.pg.PlotItem()
v.setCentralItem(plt)
plt.plot([1,4,2,3,6,2,3,4,2,3], pen='g')

if __name__ == '__main__':
    pg.exec()
