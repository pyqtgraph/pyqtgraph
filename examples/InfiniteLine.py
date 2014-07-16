# -*- coding: utf-8 -*-

"""
Example illustrating the new InfiniteLineLabel item

@author : Vincent Le Saux (vincent.le_saux@ensta-bretagne.fr)
"""
import initExample ## Add path to library (just for examples; you do not need this)
import sys
from PyQt4 import QtGui
import pyqtgraph as pg

class InfiniteLineLabelExample(QtGui.QWidget):
    
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.plot = pg.PlotWidget(name="xsection")
        data = pg.PlotDataItem([0,1,2,3],[1,2,3,4])
        self.plot.addItem(data)       
        self.plot.setTitle("InfiniteLineLabel example")
        grid = QtGui.QGridLayout(self)
        grid.addWidget(self.plot,0,0)   
        self.setLayout(grid)
        self.setWindowTitle('PyQtGraph InfiniteLineLabel example')      
        # it is still possible to use the InfiniteLine in the same way as before.
        # However, it is important to notice that the argument onlyLine=True
        # MUST be provided. Otherwise, some problems may occur. Compared to the 
        # previous item, a context menu with the possibility to manually change
        # the bounds is provided (and can be desactivated by adding the argument
        # visibleMenu=False)
        self.infiniteLineVertical = pg.InfiniteLine(pos=1.5,angle=90,
                                                    movable=True, bounds=[0.,3.],
                                                    onlyLine=True)
        self.plot.addItem(self.infiniteLineVertical)
        # it is also possible to add the InfiniteLine using the new InfiniteLineLabel
        # graphicsItem which adds some extra functionnality. Compared to the 
        # previous item, a PlotItem or PlotWidget must be provided (the item is
        # automatically added with the extra functionalities)
        self.infiniteLineHorizontal = pg.InfiniteLineLabel(self.plot.plotItem,
                                                           angle=0,
                                                           movable=True, pos=1.5,
                                                           onlyLine=False)

        # connection of the sigRemoveRequested signal to a remove slot. This
        # is only needed for the classical InfiniteLine item (the new 
        # InfiniteLineLabel has already such a method build-in)
        self.infiniteLineVertical.sigRemoveRequested.connect(self.remove)
        self.show()
        

    def remove(self):
        """ remove the self.infiniteLineVertical item """
        self.plot.removeItem(self.infiniteLineVertical)
        
        
def main():
    app = QtGui.QApplication(sys.argv)
    ex = InfiniteLineLabelExample()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main() 
