# -*- coding: utf-8 -*-
"""
Simple example demonstrating how to use the CounterSlider widget when dealing
with 3D array
"""

import initExample ## Add path to library (just for examples; you do not need this)
import sys
from PyQt4 import QtGui
import pyqtgraph as pg
import numpy as np

class CounterSliderExample(QtGui.QWidget):
    
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.vbImage = pg.ViewBox(lockAspect=1.)        
        self.plot = pg.PlotWidget(viewBox=self.vbImage)
        self.sliderCounter = pg.CounterSlider(value=0, inc=[1,5,20])
        # it is also possible to hide the Slider or the Counter
        #self.sliderCounter = CounterSliderWidget(enableSlider=False)
        #self.sliderCounter = CounterSliderWidget(enableCounter=False)
        layout = QtGui.QVBoxLayout()
        layout.setSpacing(0)
        layout.setMargin(0)        
        layout.addWidget(self.plot)
        layout.addWidget(self.sliderCounter)
        self.setLayout(layout)
        
        self.img = pg.ImageItem()
        self.data = 100.*np.random.normal(size=(200, 200, 201))
        self.plot.addItem(self.img)
        self.img.setImage(self.data[:,:,0])        
        self.sliderCounter.setRange(0,200)
        self.setWindowTitle('PyQtGraph SliderCounter example')                
        self.show() 

        self.sliderCounter.sigValueChanged.connect(self.update)
        
    def update(self, value):
        self.img.setImage(self.data[:,:,value]) 
        
        
def main():
    app = QtGui.QApplication(sys.argv)
    ex = CounterSliderExample()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main() 
