import sys

import pyqtgraph as pg

# For packages that require scipy, these may be needed: 
#   from scipy.stats import futil
#   from scipy.sparse.csgraph import _validation

pg.setConfigOption('background','w')
pg.setConfigOption('foreground','k')
app = QtWidgets.QApplication(sys.argv)

pw = pg.plot(x = [0, 1, 2, 4], y = [4, 5, 9, 6])
pw.showGrid(x=True,y=True)
text = pg.TextItem(html='<div style="text-align: center"><span style="color: #000000;"> %s</span></div>' % "here",anchor=(0.0, 0.0)) 
text.setPos(1.0, 5.0)
pw.addItem(text)
status = app.exec_()
sys.exit(status)
