# -*- coding: utf-8 -*-
"""
Special relativity simulation 



"""
import initExample ## Add path to library (just for examples; you do not need this)
import pyqtgraph as pg
from relativity import RelativityGUI

pg.mkQApp()
win = RelativityGUI()
win.setWindowTitle("Relativity!")
win.resize(1100,700)
win.show()
win.loadPreset(None, 'Twin Paradox (grid)')

if __name__ == '__main__':
    pg.mkQApp().exec_()
