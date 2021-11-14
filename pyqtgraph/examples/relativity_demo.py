"""
Special relativity simulation 
"""

from pyqtgraph.examples.relativity import RelativityGUI

import pyqtgraph as pg

pg.mkQApp()
win = RelativityGUI()
win.setWindowTitle("Relativity!")
win.resize(1100,700)
win.show()
win.loadPreset(None, 'Twin Paradox (grid)')

if __name__ == '__main__':
    pg.exec()
