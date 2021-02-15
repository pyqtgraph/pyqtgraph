# -*- coding: utf-8 -*-
"""
In this example we create a subclass of PlotCurveItem for displaying a very large 
data set from an HDF5 file that does not fit in memory. 

The basic approach is to override PlotCurveItem.viewRangeChanged such that it
displays only the visible portion of the data after downsampling. The data is downsampled
and the results are cached. Downsampling is done at scales of powers of two to reduce
computation and cache size, and is displayed with padding at the boundaries so fewer plot
updates are required. Downsampling and caching begins eagerly in a separate thread when
the item is initialized so that the data is ready (or closer to ready) when
the user needs it. This could alternatively be precomputed and stored in the HDF5 file.
Cache sizes are limited, and a relatively small amount of cached data
at the largest scales is required to fluidly navigate through large data sets.
"""

import initExample  # Add path to library (just for examples; you do not need this)

import os
import sys
import threading
from functools import lru_cache

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
        self.sampler = None
        pg.PlotCurveItem.__init__(self, *args, **kwds)
        
    def setDataSampler(self, sampler):
        self.sampler = sampler
        self.loadData()

    def viewRangeChanged(self):
        self.loadData(lazy=True)

    def loadData(self, lazy=False):
        if self.sampler is None:
            self.setData([])
            return

        vb = self.getViewBox()
        if vb is None:
            return  # no ViewBox yet

        x1, x2 = vb.viewRange()[0]
        if not lazy or not self.sampler.isRangeValid(x1, x2):
            x, y = self.sampler[x1:x2]
            self.setData(x, y)  # update the plot


class DataSampler:
    def __init__(self, x, y, sampleLimit=2500, cacheLimit=10000000, padding=0.3, eager=True):
        if x is not None and len(x) != len(y):
            raise ValueError

        self.x = x
        self.y = y
        self.sampleLimit = sampleLimit  # at most 4 times this number of samples will be plotted
        self.cacheLimit = cacheLimit  # at most 8 times this number of samples will be cached
        self.cacheSize = 0
        self.padding = padding
        self._minCacheLevel = max(2, (len(self) // self.cacheLimit).bit_length())  # n < 2 (ds < 4) has no effect
        self._maxCacheLevel = (len(self) // self.sampleLimit).bit_length()
        self._last = None
        if eager and (self._maxCacheLevel >= self._minCacheLevel):
            # spawn a thread to eagerly downsample data up to self._maxCacheLevel so
            # it will be ready for when the user zooms out
            self._t = threading.Thread(target=lambda: self._loadLevel(self._maxCacheLevel))
            self._t.start()

    def isRangeValid(self, x1, x2):
        if self._last is None:
            return False

        i11, i21, ds1, n = self._last
        i12, i22, ds2, n = self._range(x1, x2, pad=False)
        if not (i11 <= i12 <= i22 <= i21):
            return False
        i12, i22, ds2, n = self._range(x1, x2, pad=True)
        return ds1 == ds2

    def __getitem__(self, item):
        if not isinstance(item, slice) or item.step is not None:
            raise TypeError

        i1, i2, ds, n = self._last = self._range(item.start, item.stop)

        if n < 2:
            # downsampling has no effect
            y = self.y[i1:i2]
            i = np.arange(i1, i2)
        else:
            if n < self._minCacheLevel:
                y = downsample(self.y, i1, i2, ds)
            else:
                if hasattr(self, '_t'):
                    self._t.join()
                    del self._t
                y = self._loadLevel(n)[2 * i1 // ds: 2 * i2 // ds + 1]
            i = np.arange(i1, i1 + len(y) * ds // 2, ds // 2)
        print('loaded:', dict(
            range=[i1, i2],
            ds=ds,
            n=n,
            size=i2-i1,
            downsampled_size=len(y),
            cached=n >= self._minCacheLevel
        ))
        x = i if self.x is None else self.x[i]
        return x, y

    def __len__(self):
        return len(self.y)

    def _range(self, x1, x2, pad=True):
        if pad:
            pad = (x2 - x1) * self.padding
            x1, x2 = x1 - pad, x2 + pad
        if self.x is None:
            i1 = max(0, min(len(self), int(x1)))
            i2 = max(0, min(len(self), int(x2)))
        else:
            i1, i2 = np.searchsorted(self.x, [x1, x2]).tolist()
        ds = max(1, (i2 - i1) // self.sampleLimit)
        n = ds.bit_length()
        ds = 2 ** n
        i1 = (i1 // ds - 1) * ds
        i2 = (i2 // ds + 1) * ds
        i1 = max(0, min(len(self), i1))
        i2 = max(0, min(len(self), i2))
        return i1, i2, ds, n

    @lru_cache(maxsize=None)
    def _loadLevel(self, n):
        if n < self._minCacheLevel:
            raise ValueError
        elif n == self._minCacheLevel:
            out = downsample(self.y, ds=2 ** self._minCacheLevel)
        else:
            lv = self._loadLevel(n - 1)
            out = downsample(lv, ds=4)
        self.cacheSize += len(out)

        print('new cache entry:', dict(
            level=n,
            level_size=len(out),
            cache_size=self.cacheSize,
            data_size=len(self.y),
            ratio=len(self.y) // self.cacheSize
        ))
        return out


def downsample(data, start=0, stop=None, ds=4, chunksize=1000000):
    # Here convert data into a down-sampled array suitable for visualizing.
    # Must do this piecewise to limit memory usage.
    if stop is None:
        stop = len(data)
    stop = min(len(data), stop)
    samples = 1 + ((stop - start) // ds)
    visible = np.zeros(samples * 2, dtype=data.dtype)
    sourcePtr = start
    targetPtr = 0
    chunksize = (chunksize // ds) * ds

    while sourcePtr < stop:
        chunk = data[sourcePtr:min(stop, sourcePtr + chunksize)]
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
        visible[targetPtr:targetPtr + chunk.shape[0] * 2:2] = chunkMin
        visible[1 + targetPtr:1 + targetPtr + chunk.shape[0] * 2:2] = chunkMax
        targetPtr += chunk.shape[0] * 2

    return visible


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
        dlg += 1
    f.close()


if len(sys.argv) > 1:
    fileName = sys.argv[1]
else:
    fileName = 'test.hdf5'
    if not os.path.isfile(fileName):
        size, ok = QtGui.QInputDialog.getDouble(None, "Create HDF5 Dataset?", "This demo requires a large HDF5 array. To generate a file, enter the array size (in GB) and press OK.", 2.0)
        if not ok:
            sys.exit(0)
        else:
            createFile(int(size*1e9))
        # raise Exception("No suitable HDF5 file found. Use createFile() to generate an example file.")

f = h5py.File(fileName, 'r')
curve = BigDataPlot()
curve.setDataSampler(DataSampler(None, f['data']))
plt.addItem(curve)


# Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
