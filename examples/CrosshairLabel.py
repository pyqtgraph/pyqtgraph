# -*- coding: utf-8 -*-

"""
Example illustrating the new CrosshairLabel item

@author : Vincent Le Saux (vincent.le_saux@ensta-bretagne.fr)
"""
import initExample ## Add path to library (just for examples; you do not need this)
import sys
from PyQt4 import QtGui
import pyqtgraph as pg

class CrosshairLabelExample(QtGui.QWidget):
    
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.plot = pg.PlotWidget(name="xsection")
        data = pg.PlotDataItem([0,1,2,3],[1,2,3,4])
        self.plot.addItem(data)       
        self.plot.setTitle("CrosshairLabel example")
        grid = QtGui.QGridLayout(self)
        grid.addWidget(self.plot,0,0)   
        self.setLayout(grid)
        self.setWindowTitle('PyQtGraph CrosshairLabel example')      

        self.crosshair = pg.CrosshairLabel(self.plot.plotItem, movable=True, removable=True,
                                           hbounds=[None,3],vbounds=[1, None])

        self.show()


def main():
    app = QtGui.QApplication(sys.argv)
    ex = CrosshairLabelExample()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main() 
