# -*- coding: utf-8 -*-
"""
MultiPlotItem.py -  Graphics item used for displaying an array of PlotItems
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more information.
"""
from numpy import ndarray
from ..Qt import QtGui
from . import GraphicsLayout
from ..metaarray import *


__all__ = ['MultiPlotItem']
class MultiPlotItem(GraphicsLayout.GraphicsLayout):
    """
    Automatically generates a grid of plots from a multi-dimensional array
    """
    def __init__(self, *args, **kwds):
        GraphicsLayout.GraphicsLayout.__init__(self, *args, **kwds)
        self.plots = []


    def plot(self, data, **kwargs):
        #self.layout.clear()
        # extra plot parameters
        extraParams = {}

        # if 'pen' provided, add it as an extra param
        if 'pen' in kwargs:
            pen = kwargs['pen']
            if not isinstance(pen, QtGui.QPen):
                raise TypeError("Keyword argument 'pen' must be an instance of QtGui.QPen")
            extraParams['pen'] = pen


        if hasattr(data, 'implements') and data.implements('MetaArray'):
            if data.ndim != 2:
                raise Exception("MultiPlot currently only accepts 2D MetaArray.")
            ic = data.infoCopy()
            ax = 0
            for i in [0, 1]:
                if 'cols' in ic[i]:
                    ax = i
                    break
            #print "Plotting using axis %d as columns (%d plots)" % (ax, data.shape[ax])
            for i in range(data.shape[ax]):
                pi = self.addPlot()
                self.nextRow()
                sl = [slice(None)] * 2
                sl[ax] = i
                pi.plot(data[tuple(sl)], **extraParams)
                #self.layout.addItem(pi, i, 0)
                self.plots.append((pi, i, 0))
                info = ic[ax]['cols'][i]
                title = info.get('title', info.get('name', None))
                units = info.get('units', None)
                pi.setLabel('left', text=title, units=units)
            info = ic[1-ax]
            title = info.get('title', info.get('name', None))
            units = info.get('units', None)
            pi.setLabel('bottom', text=title, units=units)
        else:
            raise Exception("Data type %s not (yet?) supported for MultiPlot." % type(data))

    def close(self):
        for p in self.plots:
            p[0].close()
        self.plots = None
        self.clear()
