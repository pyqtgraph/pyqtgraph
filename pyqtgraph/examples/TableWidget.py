"""
Simple demonstration of TableWidget, which is an extension of QTableWidget
that automatically displays a variety of tabluar data formats.
"""

import numpy as np

import pyqtgraph as pg

app = pg.mkQApp("Table Widget Example")

w = pg.TableWidget()
w.show()
w.resize(500,500)
w.setWindowTitle('pyqtgraph example: TableWidget')

    
data = np.array([
    (1,   1.6,   'x'),
    (3,   5.4,   'y'),
    (8,   12.5,  'z'),
    (443, 1e-12, 'w'),
    ], dtype=[('Column 1', int), ('Column 2', float), ('Column 3', object)])
    
w.setData(data)

if __name__ == '__main__':
    pg.exec()
