# -*- coding: utf-8 -*-
"""
In this example we create a subclass of PlotCurveItem for displaying a very large 
data set from an HDF5 file that does not fit in memory. 

The basic approach is to override PlotCurveItem.viewRangeChanged such that it
displays only the visible portion of downsampled data, plus some padding to reduce the frequency
of plot updates. Further, a relatively small amount of downsampled data is precomputed and stored
in the HDF5 file, which is sufficient for navigation of the entire data set without aliasing or lag.
"""

import initExample  # Add path to library (just for examples; you do not need this)

import os
import sys

import h5py
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

pg.mkQApp()

plt = pg.plot()
plt.setWindowTitle('pyqtgraph example: HDF5 big data')
plt.enableAutoRange(False, False)
plt.setXRange(0, 500)


class BigDataPlot(pg.PlotCurveItem):
    def __init__(self, *args, **kwds):
        self._plotData = None
        pg.PlotCurveItem.__init__(self, *args, **kwds)

    def setPlotData(self, plotData):
        self._plotData = plotData
        self.replot()

    def viewRangeChanged(self):
        self.replot(lazy=True)

    def replot(self, lazy=False):
        if self._plotData is None:
            self.setData([])
            return

        vb = self.getViewBox()
        if vb is None:
            return  # no ViewBox yet

        x1, x2 = vb.viewRange()[0]
        data = self._plotData.sample(x1, x2, lazy=lazy)

        if data is not None:
            self.setData(*data)  # update the plot


class PlotData:
    def __init__(self, x, y, cache=None):
        if x is None:
            x = len(y)
        elif len(x) != len(y):
            raise ValueError

        self._x = x
        self._y = y
        self._cache = cache
        self._lastRange = None

    def sample(self, x1, x2, lazy=True, padding=0.3, **kwargs):
        i1, i2, ds = plotDataRange(self._x, x1, x2, padding=padding, **kwargs)

        if lazy and self._lastRange is not None:
            i11, i21, ds1 = self._lastRange
            i12, i22, _ = plotDataRange(self._x, x1, x2, padding=0, **kwargs)
            if (i11 <= i12 <= i22 <= i21) and (ds1 == ds):
                return

        i, y = downsample(data=self._y, start=i1, stop=i2, ds=ds, cache=self._cache)

        if isinstance(self._x, int):
            x = np.arange(i.start, i.stop, i.step)
        else:
            x = self._x[i]

        self._lastRange = (i1, i2, ds)
        return x, y


def plotDataRange(x, x1, x2, padding=0.3, sampleLimit=2500, regularize=True):
    pad = (x2 - x1) * padding
    x1, x2 = x1 - pad, x2 + pad

    if isinstance(x, int):
        n = x
        i1 = max(0, min(n, int(x1)))
        i2 = max(0, min(n, int(x2)))
    else:
        n = len(x)
        i1, i2 = np.searchsorted(x, [x1, x2]).tolist()

    ds = max(1, (i2 - i1) // sampleLimit)

    if regularize:
        lv = ds.bit_length()
        ds = 2 ** lv
        i1 = (i1 // ds - 1) * ds
        i2 = (i2 // ds + 1) * ds
        i1 = max(0, min(n, i1))
        i2 = max(0, min(n, i2))

    return i1, i2, ds


def downsample(data, start, stop, ds, cache=None, **kwargs):
    if not (0 <= start < stop <= len(data)):
        raise ValueError
    if ds <= 2:
        # downsampling has no effect
        dat = data[start:stop]
        idx = slice(start, stop)
    else:
        if cache is not None and ds in cache:
            dat = cache[ds][2 * start // ds: 2 * stop // ds + 1]
        else:
            dat = _downsample(data=data, start=start, stop=stop, ds=ds, **kwargs)
        idx = slice(start, start + len(dat) * ds // 2, ds // 2)
    return idx, dat


def _downsample(data, start, stop, ds, chunksize=1000000):
    # Here convert data into a down-sampled array suitable for visualizing.
    # Must do this piecewise to limit memory usage.

    samples = 1 + ((stop - start) // ds)
    visible = np.zeros(samples * 2, dtype=data.dtype)
    sourcePtr = start
    targetPtr = 0
    chunksize = (chunksize // ds) * ds

    while sourcePtr < stop:
        chunk = data[sourcePtr: min(stop, sourcePtr + chunksize)]
        sourcePtr += len(chunk)

        if len(chunk) % ds != 0:
            # minimally extend chunk to be integral multiple of ds in a way that doesn't affect min/max
            tail = np.full(ds - len(chunk) % ds, chunk[-1])
            chunk = np.append(chunk, tail)

        chunk = chunk.reshape(len(chunk) // ds, ds)

        # compute max and min
        chunkMax = chunk.max(axis=1)
        chunkMin = chunk.min(axis=1)

        # interleave min and max into plot data to preserve envelope shape
        visible[targetPtr: targetPtr + chunk.shape[0] * 2: 2] = chunkMin
        visible[1 + targetPtr: 1 + targetPtr + chunk.shape[0] * 2: 2] = chunkMax
        targetPtr += chunk.shape[0] * 2

    return visible


def computeDownsampleCache(data, minLevelSize=2500, maxLevelSize=10000000):
    minLevel = max(2, (len(data) // maxLevelSize).bit_length())
    maxLevel = (len(data) // minLevelSize).bit_length()

    out = {}
    dat = None
    for lv in range(minLevel, maxLevel + 1):
        ds = 2 ** lv
        if dat is None:
            dat = _downsample(data=data, start=0, stop=len(data), ds=ds)
        else:
            dat = _downsample(data=dat, start=0, stop=len(dat), ds=4)
        out[ds] = dat
    return out


def createFile(finalSize=2000000000):
    """Create a large HDF5 data file for testing.
    Data consists of 1M random samples tiled through the end of the array.
    """
    
    chunk = np.random.normal(size=1000000).astype(np.float32)
    
    f = h5py.File('test.hdf5', 'w')
    f.create_dataset('data', data=chunk, chunks=True, maxshape=(None,))
    data = f['data']

    nChunks = finalSize // (chunk.size * chunk.itemsize)
    with pg.ProgressDialog("Generating test.hdf5...", 0, nChunks) as dlg:
        for i in range(nChunks):
            newshape = [data.shape[0] + chunk.shape[0]]
            data.resize(newshape)
            data[-chunk.shape[0]:] = chunk
            dlg += 1
            if dlg.wasCanceled():
                f.close()
                os.remove('test.hdf5')
                sys.exit()

    # cache a relatively small amount of downsampled data for quick plotting
    for ds, dat in computeDownsampleCache(data).items():
        f.create_dataset(f'/ds/{ds}', data=dat)

    f.close()


if len(sys.argv) > 1:
    fileName = sys.argv[1]
else:
    fileName = 'test.hdf5'
    if not os.path.isfile(fileName):
        size, ok = QtGui.QInputDialog.getDouble(
            None,
            "Create HDF5 Dataset?",
            "This demo requires a large HDF5 array. To generate a file, enter the array size (in GB) and press OK.",
            2.0
        )
        if not ok:
            sys.exit(0)
        else:
            createFile(int(size*1e9))
        # raise Exception("No suitable HDF5 file found. Use createFile() to generate an example file.")


f = h5py.File(fileName, 'r')
x = None
y = f['data']
cache = {int(ds): dat for ds, dat in f['ds'].items()}
# cache = computeDownsampleCache(y)

curve = BigDataPlot()
curve.setPlotData(PlotData(x, y, cache=cache))
plt.addItem(curve)


# Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
