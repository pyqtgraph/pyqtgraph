# -*- coding: utf-8 -*-
"""
Simple example of subclassing GraphItem.
"""

import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

w = pg.GraphicsLayoutWidget(show=True)
w.setWindowTitle('pyqtgraph example: CustomGraphItem')
v = w.addViewBox()
v.setAspectLocked()

class Graph(pg.GraphItem):
    def __init__(self):
        self.dragPoint = None
        self.dragOffset = None
        self.textItems = []
        pg.GraphItem.__init__(self)
        self.scatter.sigClicked.connect(self.clicked)
        
    def setData(self, **kwds):
        self.text = kwds.pop('text', [])
        self.data = kwds
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        self.setTexts(self.text)
        self.updateGraph()
        
    def setTexts(self, text):
        for i in self.textItems:
            i.scene().removeItem(i)
        self.textItems = []
        for t in text:
            item = pg.TextItem(t)
            self.textItems.append(item)
            item.setParentItem(self)
        
    def updateGraph(self):
        pg.GraphItem.setData(self, **self.data)
        for i,item in enumerate(self.textItems):
            item.setPos(*self.data['pos'][i])
        
        
    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        
        if ev.isStart():
            # We are already one step into the drag.
            # Find the point(s) at the mouse cursor when the button was first 
            # pressed:
            pos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(pos)
            if len(pts) == 0:
                ev.ignore()
                return
            self.dragPoint = pts[0]
            ind = pts[0].data()[0]
            self.dragOffset = self.data['pos'][ind] - pos
        elif ev.isFinish():
            self.dragPoint = None
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return
        
        ind = self.dragPoint.data()[0]
        self.data['pos'][ind] = ev.pos() + self.dragOffset
        self.updateGraph()
        ev.accept()
        
    def clicked(self, pts):
        print("clicked: %s" % pts)


g = Graph()
v.addItem(g)

## Define positions of nodes
pos = np.array([
    [0,0],
    [10,0],
    [0,10],
    [10,10],
    [5,5],
    [15,5]
    ], dtype=float)
    
## Define the set of connections in the graph
adj = np.array([
    [0,1],
    [1,3],
    [3,2],
    [2,0],
    [1,5],
    [3,5],
    ])
    
## Define the symbol to use for each node (this is optional)
symbols = ['o','o','o','o','t','+']

## Define the line style for each connection (this is optional)
lines = np.array([
    (255,0,0,255,1),
    (255,0,255,255,2),
    (255,0,255,255,3),
    (255,255,0,255,2),
    (255,0,0,255,1),
    (255,255,255,255,4),
    ], dtype=[('red',np.ubyte),('green',np.ubyte),('blue',np.ubyte),('alpha',np.ubyte),('width',float)])

## Define text to show next to each symbol
texts = ["Point %d" % i for i in range(6)]

## Update the graph
g.setData(pos=pos, adj=adj, pen=lines, size=1, symbol=symbols, pxMode=False, text=texts)




if __name__ == '__main__':
    pg.Qt.QtWidgets.QApplication.exec_()
