# -*- coding: utf-8 -*-
"""
In this example we create a subclass of PlotCurveItem for displaying a very large 
data set from an HDF5 file that does not fit in memory. 

The basic approach is to override PlotCurveItem.viewRangeChanged such that it
reads only the portion of the HDF5 data that is necessary to display the visible
portion of the data. This is further downsampled to reduce the number of samples 
being displayed.

A more clever implementation of this class would employ some kind of caching 
to avoid re-reading the entire visible waveform at every update.
"""

import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import h5py

import sys, os
if len(sys.argv) > 1:
    fileName = sys.argv[1]
else:
    fileName = 'test.hdf5'
    if not os.path.isfile(fileName):
        print("No suitable HDF5 file found. Use createFile() to generate an example file.")
        os._exit(1)

plt = pg.plot()
plt.setWindowTitle('pyqtgraph example: HDF5 big data')
plt.enableAutoRange(False, False)
plt.setXRange(0, 500)

class HDF5Plot(pg.PlotCurveItem):
    def __init__(self, *args, **kwds):
        self.hdf5 = None
        self.limit = 10000 # maximum number of samples to be plotted
        pg.PlotCurveItem.__init__(self, *args, **kwds)
        
    def setHDF5(self, data):
        self.hdf5 = data
        self.updateHDF5Plot()
        
    def viewRangeChanged(self):
        self.updateHDF5Plot()
        
    def updateHDF5Plot(self):
        if self.hdf5 is None:
            self.setData([])
            return
        
        vb = self.getViewBox()
        if vb is None:
            return  # no ViewBox yet
        
        # Determine what data range must be read from HDF5
        xrange = vb.viewRange()[0]
        start = max(0,int(xrange[0])-1)
        stop = min(len(self.hdf5), int(xrange[1]+2))
        
        # Decide by how much we should downsample 
        ds = int((stop-start) / self.limit) + 1
        
        if ds == 1:
            # Small enough to display with no intervention.
            visible = self.hdf5[start:stop]
            scale = 1
        else:
            # Here convert data into a down-sampled array suitable for visualizing.
            # Must do this piecewise to limit memory usage.        
            samples = 1 + ((stop-start) // ds)
            visible = np.zeros(samples*2, dtype=self.hdf5.dtype)
            sourcePtr = start
            targetPtr = 0
            
            # read data in chunks of ~1M samples
            chunkSize = (1000000//ds) * ds
            while sourcePtr < stop-1: 
                chunk = self.hdf5[sourcePtr:min(stop,sourcePtr+chunkSize)]
                sourcePtr += len(chunk)
                
                # reshape chunk to be integral multiple of ds
                chunk = chunk[:(len(chunk)//ds) * ds].reshape(len(chunk)//ds, ds)
                
                # compute max and min
                chunkMax = chunk.max(axis=1)
                chunkMin = chunk.min(axis=1)
                
                # interleave min and max into plot data to preserve envelope shape
                visible[targetPtr:targetPtr+chunk.shape[0]*2:2] = chunkMin
                visible[1+targetPtr:1+targetPtr+chunk.shape[0]*2:2] = chunkMax
                targetPtr += chunk.shape[0]*2
            
            visible = visible[:targetPtr]
            scale = ds * 0.5
            
        self.setData(visible) # update the plot
        self.setPos(start, 0) # shift to match starting index
        self.resetTransform()
        self.scale(scale, 1)  # scale to match downsampling

        
f = h5py.File(fileName, 'r')
curve = HDF5Plot()
curve.setHDF5(f['data'])
plt.addItem(curve)


def createFile(finalSize=2000000000):
    """Create a large HDF5 data file for testing.
    Data consists of 1M random samples tiled through the end of the array.
    """
    
    chunk = np.random.normal(size=1000000).astype(np.float32)
    
    f = h5py.File('test.hdf5', 'w')
    f.create_dataset('data', data=chunk, chunks=True, maxshape=(None,))
    data = f['data']

    for i in range(finalSize // (chunk.size * chunk.itemsize)):
        newshape = [data.shape[0] + chunk.shape[0]]
        data.resize(newshape)
        data[-chunk.shape[0]:] = chunk
        
    f.close()
    


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()




