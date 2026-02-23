'''
Demonstrate usage of BoxplotItem
'''

import numpy as np
import pyqtgraph as pg

np.random.seed(8)
n = 5
data = [np.random.normal(500, 30, 1000) for _ in range(n)]

win = pg.GraphicsLayoutWidget(show=True)
win.setWindowTitle('pyqtgraph example: BoxplotItem')
win.resize(1000, 800)

p1 = win.addPlot(title="1.5IQR whiskers")
bp1 = pg.BoxplotItem()
# `data` must be a 2d array or a list of 1d arraylike
# if your data is 1d, put it into a list first
bp1.setData(data=data, symbolBrush="g")
p1.addItem(bp1)

# user can define their own whisker boundaries 
# here is the example uses minimum and maximum value
# as lower and upper whisker boundaries
def min_max_whisker(d):
    # `d` is a 1d arraylike
    # returns (lower, upper)
    return min(d), max(d)

p2 = win.addPlot(title="Use min and max as whiskers")
bp2 = pg.BoxplotItem()
bp2.setData(data=data, symbolBrush="g")
bp2.setWhiskerFunc(min_max_whisker)
p2.addItem(bp2)

win.nextRow()

p3 = win.addPlot(title="Horizontal boxplot")
bp3 = pg.BoxplotItem()
bp3.setData(data=data, locAsX=False, symbol="star", symbolBrush="g")
p3.addItem(bp3)

p4 = win.addPlot(title="Customize box positions")
bp4 = pg.BoxplotItem()
bp4.setData(loc=[i*(i+1)/2 for i in range(n)], data=data, locAsX=False, symbol="star", symbolBrush="g")
p4.addItem(bp4)

if __name__ == '__main__':
    pg.exec()