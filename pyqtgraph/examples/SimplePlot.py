import pyqtgraph as pg
import pyqtgraph.exporters
import numpy as np
plt = pg.plot(np.random.normal(size=100), title="Simplest possible plotting example")

if __name__ == '__main__':
    pg.exec()
