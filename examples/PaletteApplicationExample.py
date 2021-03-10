#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Update a simple plot as rapidly as possible to measure speed.
"""

## Add path to library (just for examples; you do not need this)
import initExample

import numpy as np

from pyqtgraph.Qt import mkQApp, QtCore, QtWidgets
from pyqtgraph.ptime import time
import pyqtgraph as pg


class MainWindow(QtWidgets.QMainWindow):
    """ example application main window """
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        main_wid = QtWidgets.QWidget()
        self.setCentralWidget(main_wid)
        self.setWindowTitle('pyqtgraph example: Palette application test')
        self.resize(600,600)
        
        pg.palette.get('monogreen').apply()

        main_layout = QtWidgets.QGridLayout( main_wid )
        gr_wid = pg.GraphicsLayoutWidget(show=True)
        main_layout.addWidget( gr_wid, 0,0, 1,4 )

        btn = QtWidgets.QPushButton('continuous')
        btn.clicked.connect(self.handle_button_timer_on)
        main_layout.addWidget(btn, 1,0, 1,1 )

        btn = QtWidgets.QPushButton('stop updates')
        btn.clicked.connect(self.handle_button_timer_off)
        main_layout.addWidget(btn, 2,0, 1,1 )
        
        btn = QtWidgets.QPushButton('apply <legacy>')
        btn.clicked.connect(self.handle_button_pal1)
        main_layout.addWidget(btn, 1,2, 1,1 )

        btn = QtWidgets.QPushButton('apply <mono green>')
        btn.clicked.connect(self.handle_button_pal2)
        main_layout.addWidget(btn, 1,3, 1,1 )

        btn = QtWidgets.QPushButton('apply <relaxed - dark>')
        btn.clicked.connect(self.handle_button_pal3)
        main_layout.addWidget(btn, 2,2, 1,1 )

        btn = QtWidgets.QPushButton('apply <relaxed - light>')
        btn.clicked.connect(self.handle_button_pal4)
        main_layout.addWidget(btn, 2,3, 1,1 )

        btn = QtWidgets.QPushButton('legacy fg/bg override 1')
        btn.clicked.connect(self.handle_button_leg1)
        main_layout.addWidget(btn, 3,2, 1,1 )

        btn = QtWidgets.QPushButton('legacy fg/bg override 2')
        btn.clicked.connect(self.handle_button_leg2)
        main_layout.addWidget(btn, 3,3, 1,1 )

        self.plt = gr_wid.addPlot()
        self.plt.enableAutoRange(False)
        self.plt.setYRange( -7,7 )
        self.plt.setXRange( 0, 15 ) #500 )
    
        self.plt.setLabel('bottom', 'Index', units='B')

        self.data1 = +3 + np.random.normal(size=(15)) #500))
        self.data2 = -3 + np.random.normal(size=(15)) #500))

        self.curve1 = pg.PlotDataItem(pen='r', symbol='o', symbolSize=10, symbolPen='gr_fg', symbolBrush=('y',127))
        self.plt.addItem(self.curve1)
        
        self.curve2 = pg.PlotCurveItem(pen='p3', brush='p4')
        self.curve2.setFillLevel(0)
        self.plt.addItem(self.curve2)
        self.show()
        
        self.pal_1 = pg.palette.get('legacy')
        self.pal_2 = pg.palette.get('monogreen')
        self.pal_3 = pg.palette.get('relaxed_dark')
        self.pal_4 = pg.palette.get('relaxed_light')

        self.lastTime = time()
        self.fps = None
        self.timer = QtCore.QTimer(singleShot=False)
        self.timer.timeout.connect( self.timed_update )
        
        self.timed_update()
    
    def testSignal(self, val):
        """ demonstrate use of PaletteChanged signal """
        print('"Palette changed" signal was received with value', val)
        

    def handle_button_timer_on(self):
        """ (re-)activate timer """
        self.timer.start(1)

    def handle_button_timer_off(self):
        """ de-activate timer """
        self.timer.stop()

    def handle_button_pal1(self):
        """ apply palette 1 on request """
        print('--> legacy')
        self.pal_1.apply()

    def handle_button_pal2(self):
        """ apply palette 2 on request """
        print('--> mono green')
        self.pal_2.apply()

    def handle_button_pal3(self):
        """ apply palette 1 on request """
        print('--> relax(light)')
        self.pal_3.apply()

    def handle_button_pal4(self):
        """ apply palette 1 on request """
        print('--> relax(light)')
        self.pal_4.apply()
        
    def handle_button_leg1(self):
        """ test legacy background / foreground overrides """
        pg.setConfigOption('background', '#ff0000')
        pg.setConfigOption('foreground', '#0000ff')

    def handle_button_leg2(self):
        """ test legacy background / foreground overrides """
        pg.setConfigOption('background', '#0000ff')
        pg.setConfigOption('foreground', '#ff0000')

    def timed_update(self):
        """ update loop, called by timer """
        self.data1[:-1] = self.data1[1:]
        self.data1[-1] = +3 + np.random.normal()
        xdata = np.arange( len(self.data1) )
        self.curve1.setData( x=xdata, y=self.data1 )

        self.data2[:-1] = self.data2[1:]
        self.data2[-1] = -3 + np.random.normal()
        self.curve2.setData( y=self.data2 )

        now = time()
        dt = now - self.lastTime
        self.lastTime = now
        if self.fps is None:
            self.fps = 1.0/dt
        else:
            s = np.clip(dt*3., 0, 1)
            self.fps = self.fps * (1-s) + (1.0/dt) * s
        self.plt.setTitle('%0.2f fps' % self.fps)
        QtWidgets.QApplication.processEvents()  ## force complete redraw for every plot

mkQApp("Palette test application")
main_window = MainWindow()

## Start Qt event loop
if __name__ == '__main__':
    QtWidgets.QApplication.instance().exec_()
