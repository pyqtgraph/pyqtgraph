import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
import numpy as np
plt = pg.plot(np.random.normal(size=100), title="Simplest possible plotting example")

if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1 or not hasattr(pg.QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec_()
