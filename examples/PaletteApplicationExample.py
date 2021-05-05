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
        
        test_palette = pg.palette.get('system')
        
        pg.palette.get('relaxed_dark').apply()

        main_layout = QtWidgets.QGridLayout( main_wid )
        gr_wid = pg.GraphicsLayoutWidget(show=True)
        main_layout.addWidget( gr_wid, 0,0, 1,5 )

        btn = QtWidgets.QPushButton('continuous')
        btn.clicked.connect(self.handle_button_timer_on)
        main_layout.addWidget(btn, 1,0, 1,1 )

        btn = QtWidgets.QPushButton('stop updates')
        btn.clicked.connect(self.handle_button_timer_off)
        main_layout.addWidget(btn, 2,0, 1,1 )
        
        palette_buttons = (
            ('apply <legacy>', 1,2, self.handle_button_pal1 ),
            ('legacy fg/bg 1', 1,3, self.handle_button_leg1 ),
            ('legacy fg/bg 2', 1,4, self.handle_button_leg2 ),
            ('apply <mono green>', 2,2, self.handle_button_mono1 ),
            ('apply <mono amber>', 2,3, self.handle_button_mono2 ),
            ('apply <mono blue>' , 2,4, self.handle_button_mono3 ),
            ('apply <relaxed-dark>' , 3,2, self.handle_button_pal2 ),
            ('apply <relaxed-light>', 3,3, self.handle_button_pal3 )
        )
        for text, row, col, func in palette_buttons:
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(func)
            main_layout.addWidget(btn, row,col, 1,1 )

        self.plt = gr_wid.addPlot()
        self.plt.enableAutoRange(False)
        self.plt.setYRange( -7,7 )
        self.plt.setXRange( 0, 15 ) #500 )
    
        self.plt.setLabel('bottom', 'Index', units='B')

        self.data1 = +3 + np.random.normal(size=(15)) #500))
        self.data2 = -3 + np.random.normal(size=(15)) #500))

        # self.curve1 = pg.PlotDataItem(
        #     pen='r', 
        #     symbol='o', symbolSize=10, symbolPen='gr_fg', symbolBrush=('y',127), 
        #     hoverable=True, hoverPen='w', hoverBrush='w')
        self.curve1 = pg.ScatterPlotItem(
            symbol='o', symbolSize=12, symbolPen='gr_fg', symbolBrush=('y',127), 
            hoverable=True, hoverPen='gr_acc', hoverBrush='gr_reg')
        # self.curve1.setHoverable(True)
        self.plt.addItem(self.curve1)
        
        self.curve2 = pg.PlotCurveItem(pen='l', brush='d')
        self.curve2.setFillLevel(0)
        self.plt.addItem(self.curve2)
        self.show()
        
        self.pal_1 = pg.palette.get('legacy')
        self.pal_2 = pg.palette.get('relaxed_dark')
        self.pal_3 = pg.palette.get('relaxed_light')
        self.mpal_1 = pg.palette.make_monochrome('green')
        self.mpal_2 = pg.palette.make_monochrome('amber')
        self.mpal_3 = pg.palette.make_monochrome('blue')

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
        self.pal_1.apply()

    def handle_button_pal2(self):
        """ apply palette 2 on request """
        self.pal_2.apply()

    def handle_button_pal3(self):
        """ apply palette 1 on request """
        self.pal_3.apply()

    def handle_button_mono1(self):
        """ apply monochrome palette 1 on request """
        self.mpal_1.apply()

    def handle_button_mono2(self):
        """ apply monochrome palette 2 on request """
        self.mpal_2.apply()

    def handle_button_mono3(self):
        """ apply monochrome palette 3 on request """
        self.mpal_3.apply()

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
