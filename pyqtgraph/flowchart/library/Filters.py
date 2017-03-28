# -*- coding: utf-8 -*-
import numpy as np
from ...Qt import QtCore, QtGui
from ..Node import Node
from . import functions
from ... import functions as pgfn
from .common import *
from ...python2_3 import xrange
from ... import PolyLineROI
from ... import Point
from ... import metaarray as metaarray


class Downsample(CtrlNode):
    """Downsample by averaging samples together."""
    nodeName = 'Downsample'
    uiTemplate = [
        ('n', 'intSpin', {'min': 1, 'max': 1000000})
    ]
    
    def processData(self, data):
        return functions.downsample(data, self.ctrls['n'].value(), axis=0)


class Subsample(CtrlNode):
    """Downsample by selecting every Nth sample."""
    nodeName = 'Subsample'
    uiTemplate = [
        ('n', 'intSpin', {'min': 1, 'max': 1000000})
    ]
    
    def processData(self, data):
        return data[::self.ctrls['n'].value()]


class Bessel(CtrlNode):
    """Bessel filter. Input data must have time values."""
    nodeName = 'BesselFilter'
    uiTemplate = [
        ('band', 'combo', {'values': ['lowpass', 'highpass'], 'index': 0}),
        ('cutoff', 'spin', {'value': 1000., 'step': 1, 'dec': True, 'range': [0.0, None], 'suffix': 'Hz', 'siPrefix': True}),
        ('order', 'intSpin', {'value': 4, 'min': 1, 'max': 16}),
        ('bidir', 'check', {'checked': True})
    ]
    
    def processData(self, data):
        s = self.stateGroup.state()
        if s['band'] == 'lowpass':
            mode = 'low'
        else:
            mode = 'high'
        return functions.besselFilter(data, bidir=s['bidir'], btype=mode, cutoff=s['cutoff'], order=s['order'])


class Butterworth(CtrlNode):
    """Butterworth filter"""
    nodeName = 'ButterworthFilter'
    uiTemplate = [
        ('band', 'combo', {'values': ['lowpass', 'highpass'], 'index': 0}),
        ('wPass', 'spin', {'value': 1000., 'step': 1, 'dec': True, 'range': [0.0, None], 'suffix': 'Hz', 'siPrefix': True}),
        ('wStop', 'spin', {'value': 2000., 'step': 1, 'dec': True, 'range': [0.0, None], 'suffix': 'Hz', 'siPrefix': True}),
        ('gPass', 'spin', {'value': 2.0, 'step': 1, 'dec': True, 'range': [0.0, None], 'suffix': 'dB', 'siPrefix': True}),
        ('gStop', 'spin', {'value': 20.0, 'step': 1, 'dec': True, 'range': [0.0, None], 'suffix': 'dB', 'siPrefix': True}),
        ('bidir', 'check', {'checked': True})
    ]
    
    def processData(self, data):
        s = self.stateGroup.state()
        if s['band'] == 'lowpass':
            mode = 'low'
        else:
            mode = 'high'
        ret = functions.butterworthFilter(data, bidir=s['bidir'], btype=mode, wPass=s['wPass'], wStop=s['wStop'], gPass=s['gPass'], gStop=s['gStop'])
        return ret

        
class ButterworthNotch(CtrlNode):
    """Butterworth notch filter"""
    nodeName = 'ButterworthNotchFilter'
    uiTemplate = [
        ('low_wPass', 'spin', {'value': 1000., 'step': 1, 'dec': True, 'range': [0.0, None], 'suffix': 'Hz', 'siPrefix': True}),
        ('low_wStop', 'spin', {'value': 2000., 'step': 1, 'dec': True, 'range': [0.0, None], 'suffix': 'Hz', 'siPrefix': True}),
        ('low_gPass', 'spin', {'value': 2.0, 'step': 1, 'dec': True, 'range': [0.0, None], 'suffix': 'dB', 'siPrefix': True}),
        ('low_gStop', 'spin', {'value': 20.0, 'step': 1, 'dec': True, 'range': [0.0, None], 'suffix': 'dB', 'siPrefix': True}),
        ('high_wPass', 'spin', {'value': 3000., 'step': 1, 'dec': True, 'range': [0.0, None], 'suffix': 'Hz', 'siPrefix': True}),
        ('high_wStop', 'spin', {'value': 4000., 'step': 1, 'dec': True, 'range': [0.0, None], 'suffix': 'Hz', 'siPrefix': True}),
        ('high_gPass', 'spin', {'value': 2.0, 'step': 1, 'dec': True, 'range': [0.0, None], 'suffix': 'dB', 'siPrefix': True}),
        ('high_gStop', 'spin', {'value': 20.0, 'step': 1, 'dec': True, 'range': [0.0, None], 'suffix': 'dB', 'siPrefix': True}),
        ('bidir', 'check', {'checked': True})
    ]
    
    def processData(self, data):
        s = self.stateGroup.state()
        
        low = functions.butterworthFilter(data, bidir=s['bidir'], btype='low', wPass=s['low_wPass'], wStop=s['low_wStop'], gPass=s['low_gPass'], gStop=s['low_gStop'])
        high = functions.butterworthFilter(data, bidir=s['bidir'], btype='high', wPass=s['high_wPass'], wStop=s['high_wStop'], gPass=s['high_gPass'], gStop=s['high_gStop'])
        return low + high
    

class Mean(CtrlNode):
    """Filters data by taking the mean of a sliding window"""
    nodeName = 'MeanFilter'
    uiTemplate = [
        ('n', 'intSpin', {'min': 1, 'max': 1000000})
    ]
    
    @metaArrayWrapper
    def processData(self, data):
        n = self.ctrls['n'].value()
        return functions.rollingSum(data, n) / n


class Median(CtrlNode):
    """Filters data by taking the median of a sliding window"""
    nodeName = 'MedianFilter'
    uiTemplate = [
        ('n', 'intSpin', {'min': 1, 'max': 1000000})
    ]
    
    @metaArrayWrapper
    def processData(self, data):
        try:
            import scipy.ndimage
        except ImportError:
            raise Exception("MedianFilter node requires the package scipy.ndimage.")
        return scipy.ndimage.median_filter(data, self.ctrls['n'].value())

class Mode(CtrlNode):
    """Filters data by taking the mode (histogram-based) of a sliding window"""
    nodeName = 'ModeFilter'
    uiTemplate = [
        ('window', 'intSpin', {'value': 500, 'min': 1, 'max': 1000000}),
    ]
    
    @metaArrayWrapper
    def processData(self, data):
        return functions.modeFilter(data, self.ctrls['window'].value())


class Denoise(CtrlNode):
    """Removes anomalous spikes from data, replacing with nearby values"""
    nodeName = 'DenoiseFilter'
    uiTemplate = [
        ('radius', 'intSpin', {'value': 2, 'min': 0, 'max': 1000000}),
        ('threshold', 'doubleSpin', {'value': 4.0, 'min': 0, 'max': 1000})
    ]
    
    def processData(self, data):
        #print "DENOISE"
        s = self.stateGroup.state()
        return functions.denoise(data, **s)


class Gaussian(CtrlNode):
    """Gaussian smoothing filter."""
    nodeName = 'GaussianFilter'
    uiTemplate = [
        ('sigma', 'doubleSpin', {'min': 0, 'max': 1000000})
    ]
    
    @metaArrayWrapper
    def processData(self, data):
        try:
            import scipy.ndimage
        except ImportError:
            raise Exception("GaussianFilter node requires the package scipy.ndimage.")

        if hasattr(data, 'implements') and data.implements('MetaArray'):
            info = data.infoCopy()
            filt = pgfn.gaussianFilter(data.asarray(), self.ctrls['sigma'].value())
            if 'values' in info[0]:
                info[0]['values'] = info[0]['values'][:filt.shape[0]]
            return metaarray.MetaArray(filt, info=info)
        else:
            return pgfn.gaussianFilter(data, self.ctrls['sigma'].value())

class Derivative(CtrlNode):
    """Returns the pointwise derivative of the input"""
    nodeName = 'DerivativeFilter'
    
    def processData(self, data):
        if hasattr(data, 'implements') and data.implements('MetaArray'):
            info = data.infoCopy()
            if 'values' in info[0]:
                info[0]['values'] = info[0]['values'][:-1]
            return metaarray.MetaArray(data[1:] - data[:-1], info=info)
        else:
            return data[1:] - data[:-1]


class Integral(CtrlNode):
    """Returns the pointwise integral of the input"""
    nodeName = 'IntegralFilter'
    
    @metaArrayWrapper
    def processData(self, data):
        data[1:] += data[:-1]
        return data


class Detrend(CtrlNode):
    """Removes linear trend from the data"""
    nodeName = 'DetrendFilter'
    
    @metaArrayWrapper
    def processData(self, data):
        try:
            from scipy.signal import detrend
        except ImportError:
            raise Exception("DetrendFilter node requires the package scipy.signal.")
        return detrend(data)

class RemoveBaseline(PlottingCtrlNode):
    """Remove an arbitrary, graphically defined baseline from the data."""
    nodeName = 'RemoveBaseline'
    
    def __init__(self, name):
        ## define inputs and outputs (one output needs to be a plot)
        PlottingCtrlNode.__init__(self, name)
        self.line = PolyLineROI([[0,0],[1,0]])
        self.line.sigRegionChanged.connect(self.changed)
        
        ## create a PolyLineROI, add it to a plot -- actually, I think we want to do this after the node is connected to a plot (look at EventDetection.ThresholdEvents node for ideas), and possible after there is data. We will need to update the end positions of the line each time the input data changes
        #self.line = None ## will become a PolyLineROI
        
    def connectToPlot(self, node):
        """Define what happens when the node is connected to a plot"""

        if node.plot is None:
            return
        node.getPlot().addItem(self.line)
       
    def disconnectFromPlot(self, plot):
        """Define what happens when the node is disconnected from a plot"""
        plot.removeItem(self.line)    
    
    def processData(self, data):
        ## get array of baseline (from PolyLineROI)
        h0 = self.line.getHandles()[0]
        h1 = self.line.getHandles()[-1]
        
        timeVals = data.xvals(0)
        h0.setPos(timeVals[0], h0.pos()[1])
        h1.setPos(timeVals[-1], h1.pos()[1])      
        
        pts = self.line.listPoints() ## lists line handles in same coordinates as data
        pts, indices = self.adjustXPositions(pts, timeVals) ## maxe sure x positions match x positions of data points
        
        ## construct an array that represents the baseline
        arr = np.zeros(len(data), dtype=float)
        n = 1
        arr[0] = pts[0].y()
        for i in range(len(pts)-1):
            x1 = pts[i].x()
            x2 = pts[i+1].x()
            y1 = pts[i].y()
            y2 = pts[i+1].y()
            m = (y2-y1)/(x2-x1)
            b = y1
            
            times = timeVals[(timeVals > x1)*(timeVals <= x2)]
            arr[n:n+len(times)] = (m*(times-times[0]))+b
            n += len(times)
                
        return data - arr ## subract baseline from data
        
    def adjustXPositions(self, pts, data):
        """Return a list of Point() where the x position is set to the nearest x value in *data* for each point in *pts*."""
        points = []
        timeIndices = []
        for p in pts:
            x = np.argwhere(abs(data - p.x()) == abs(data - p.x()).min())
            points.append(Point(data[x], p.y()))
            timeIndices.append(x)
            
        return points, timeIndices



class AdaptiveDetrend(CtrlNode):
    """Removes baseline from data, ignoring anomalous events"""
    nodeName = 'AdaptiveDetrend'
    uiTemplate = [
        ('threshold', 'doubleSpin', {'value': 3.0, 'min': 0, 'max': 1000000})
    ]
    
    def processData(self, data):
        return functions.adaptiveDetrend(data, threshold=self.ctrls['threshold'].value())

class HistogramDetrend(CtrlNode):
    """Removes baseline from data by computing mode (from histogram) of beginning and end of data."""
    nodeName = 'HistogramDetrend'
    uiTemplate = [
        ('windowSize', 'intSpin', {'value': 500, 'min': 10, 'max': 1000000, 'suffix': 'pts'}),
        ('numBins', 'intSpin', {'value': 50, 'min': 3, 'max': 1000000}),
        ('offsetOnly', 'check', {'checked': False}),
    ]
    
    def processData(self, data):
        s = self.stateGroup.state()
        #ws = self.ctrls['windowSize'].value()
        #bn = self.ctrls['numBins'].value()
        #offset = self.ctrls['offsetOnly'].checked()
        return functions.histogramDetrend(data, window=s['windowSize'], bins=s['numBins'], offsetOnly=s['offsetOnly'])


    
class RemovePeriodic(CtrlNode):
    nodeName = 'RemovePeriodic'
    uiTemplate = [
        #('windowSize', 'intSpin', {'value': 500, 'min': 10, 'max': 1000000, 'suffix': 'pts'}),
        #('numBins', 'intSpin', {'value': 50, 'min': 3, 'max': 1000000})
        ('f0', 'spin', {'value': 60, 'suffix': 'Hz', 'siPrefix': True, 'min': 0, 'max': None}),
        ('harmonics', 'intSpin', {'value': 30, 'min': 0}),
        ('samples', 'intSpin', {'value': 1, 'min': 1}),
    ]

    def processData(self, data):
        times = data.xvals('Time')
        dt = times[1]-times[0]
        
        data1 = data.asarray()
        ft = np.fft.fft(data1)
        
        ## determine frequencies in fft data
        df = 1.0 / (len(data1) * dt)
        freqs = np.linspace(0.0, (len(ft)-1) * df, len(ft))
        
        ## flatten spikes at f0 and harmonics
        f0 = self.ctrls['f0'].value()
        for i in xrange(1, self.ctrls['harmonics'].value()+2):
            f = f0 * i # target frequency
            
            ## determine index range to check for this frequency
            ind1 = int(np.floor(f / df))
            ind2 = int(np.ceil(f / df)) + (self.ctrls['samples'].value()-1)
            if ind1 > len(ft)/2.:
                break
            mag = (abs(ft[ind1-1]) + abs(ft[ind2+1])) * 0.5
            for j in range(ind1, ind2+1):
                phase = np.angle(ft[j])   ## Must preserve the phase of each point, otherwise any transients in the trace might lead to large artifacts.
                re = mag * np.cos(phase)
                im = mag * np.sin(phase)
                ft[j] = re + im*1j
                ft[len(ft)-j] = re - im*1j
                
        data2 = np.fft.ifft(ft).real
        
        ma = metaarray.MetaArray(data2, info=data.infoCopy())
        return ma
        
        
        
