from time import perf_counter

import numpy as np

import pyqtgraph as pg

app = pg.mkQApp()
plt = pg.PlotWidget()

app.processEvents()

## Putting this at the beginning or end does not have much effect
plt.show()   

## The auto-range is recomputed after each item is added,
## so disabling it before plotting helps
plt.enableAutoRange(False, False)

def plot():
    start = perf_counter()
    n = 15
    pts = 100
    x = np.linspace(0, 0.8, pts)
    y = np.random.random(size=pts)*0.8
    for i in range(n):
        for j in range(n):
            ## calling PlotWidget.plot() generates a PlotDataItem, which 
            ## has a bit more overhead than PlotCurveItem, which is all 
            ## we need here. This overhead adds up quickly and makes a big
            ## difference in speed.
            
            plt.addItem(pg.PlotCurveItem(x=x+i, y=y+j))
            
    dt = perf_counter() - start
    print(f"Create plots took: {dt * 1000:.3f} ms")

## Plot and clear 5 times, printing the time it took
for _ in range(5):
    plt.clear()
    plot()
    app.processEvents()
    plt.autoRange()





def fastPlot():
    ## Different approach:  generate a single item with all data points.
    ## This runs many times faster.
    start = perf_counter()
    n = 15
    pts = 100
    x = np.linspace(0, 0.8, pts)
    y = np.random.random(size=pts)*0.8
    shape = (n, n, pts)
    xdata = np.empty(shape)
    xdata[:] = x + np.arange(shape[1]).reshape((1,-1,1))
    ydata = np.empty(shape)
    ydata[:] = y + np.arange(shape[0]).reshape((-1,1,1))
    conn = np.ones(shape, dtype=bool)
    conn[...,-1] = False # make sure plots are disconnected
    item = pg.PlotCurveItem()
    item.setData(xdata.ravel(), ydata.ravel(), connect=conn.ravel())
    plt.addItem(item)
    
    dt = perf_counter() - start
    print("Create plots took: %0.3fms" % (dt*1000))


## Plot and clear 5 times, printing the time it took
for _ in range(5):
    plt.clear()
    fastPlot()
    app.processEvents()
    plt.autoRange()

if __name__ == '__main__':
    pg.exec()
