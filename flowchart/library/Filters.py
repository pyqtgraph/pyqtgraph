# -*- coding: utf-8 -*-
from pyqtgraph.Qt import QtCore, QtGui
from ..Node import Node
from scipy.signal import detrend
from scipy.ndimage import median_filter, gaussian_filter
#from pyqtgraph.SignalProxy import SignalProxy
from . import functions
from .common import *
import numpy as np

try:
    import metaarray
    HAVE_METAARRAY = True
except:
    HAVE_METAARRAY = False


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
        return median_filter(data, self.ctrls['n'].value())

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
        return gaussian_filter(data, self.ctrls['sigma'].value())


class Derivative(CtrlNode):
    """Returns the pointwise derivative of the input"""
    nodeName = 'DerivativeFilter'
    
    def processData(self, data):
        if HAVE_METAARRAY and isinstance(data, metaarray.MetaArray):
            info = data.infoCopy()
            if 'values' in info[0]:
                info[0]['values'] = info[0]['values'][:-1]
            return MetaArray(data[1:] - data[:-1], info=info)
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
        return detrend(data)


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
        ('numBins', 'intSpin', {'value': 50, 'min': 3, 'max': 1000000})
    ]
    
    def processData(self, data):
        ws = self.ctrls['windowSize'].value()
        bn = self.ctrls['numBins'].value()
        return functions.histogramDetrend(data, window=ws, bins=bn)


    
