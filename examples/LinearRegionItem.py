# -*- coding: utf-8 -*-

"""
Example illustrating the new LinearRegionItem item

@author : Vincent Le Saux (vincent.le_saux@ensta-bretagne.fr)
"""
import initExample ## Add path to library (just for examples; you do not need this)
import sys
from PyQt4 import QtGui
import pyqtgraph as pg

class LinearRegionItemLabelExample(QtGui.QWidget):
    
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.plot = pg.PlotWidget(name="xsection")
        data = pg.PlotDataItem([0,1,2,3],[1,2,3,4])
        self.plot.addItem(data)       
        self.plot.setTitle("LinearRegionItemLabel example")
        grid = QtGui.QGridLayout(self)
        grid.addWidget(self.plot,0,0)   
        self.setLayout(grid)
        self.setWindowTitle('PyQtGraph LinearRegionItemLabel example')   
        # it is still possible to use the LinearRegionitem in the same way as before.
        # However, it is important to notice that the argument onlyLines=True
        # MUST be provided. Otherwise, some problems may occur. Compared to the 
        # previous item, a context menu with the possibility to manually change
        # the bounds is provided (and can be desactivated by adding the argument
        # visibleMenu=False) 
        self.linearRegionItem = pg.LinearRegionItem(orientation=pg.LinearRegionItem.Vertical,
                                                         onlyLines=True)
        self.plot.addItem(self.linearRegionItem)

        self.linearRegionItemLabel = pg.LinearRegionItemLabel(self.plot.plotItem,
                                                              orientation=pg.LinearRegionItem.Vertical,
                                                              values=[1.5,2.5],
                                                              onlyLines=False)
        # connection of the sigRemoveRequested signal to a remove slot. This
        # is only needed for the classical LinearRegionItem item (the new 
        # LinearRegionItemLabel has already such a method build-in)
        self.linearRegionItem.sigRemoveRequested.connect(self.remove)
        self.show()
        
    def remove(self):
        self.plot.removeItem(self.linearRegionItem)
        
def main():
    
    app = QtGui.QApplication(sys.argv)
    ex = LinearRegionItemLabelExample()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main() 
