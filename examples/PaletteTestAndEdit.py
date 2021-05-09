#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Update a simple plot as rapidly as possible to measure speed.
"""

## Add path to library (just for examples; you do not need this)
import initExample

import qdarkstyle
import numpy as np

from pyqtgraph.Qt import mkQApp, QtCore, QtGui, QtWidgets
from pyqtgraph.ptime import time
import pyqtgraph as pg


class MainWindow(QtWidgets.QMainWindow):
    """ example application main window """
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        
        force_dark = True # start in dark mode on Windows

        self.setWindowTitle('pyqtgraph example: Palette application test')
        self.resize(600,600)

        self.palette_options = (
            ('system', 'system', []),
            ('legacy', 'legacy', []),
            ('relaxed (dark)' , 'relaxed_dark',  []),
            ('relaxed (light)', 'relaxed_light', []),
            ('pastels (light)', 'pastels', []),
            ('mono green', 'monochrome', ['green']),
            ('mono amber', 'monochrome', ['amber']),
            ('mono blue' , 'monochrome', ['blue' ]),
            ('synthwave' , 'synthwave', []),
        )

        self.colormap_options = (
            'CET-C1', 'CET-C2','CET-C6','CET-C7', 'CET-R2', 'CET-R4',
            'CET-L8', 'CET-L16', 'none'
            # , 'none', 'CET-C1', 'CET-C2', 'CET-C3', 'CET-C4', 'CET-C5', 'CET-C6', 'CET-C7', 'CET-CBC1', 'CET-CBC2'
        )

        app = QtWidgets.QApplication.instance()
        self.q_palette = {
            'system' : app.palette(),
            'dark'   : self.make_dark_QPalette()
        }
        app.setStyle("Fusion")
        
        self.ui = self.prepare_ui() # relocate long-winded window layout
        # dictionary self.ui contains references to:
        # 'sample_start' QLineEdit for start of colormap sampling
        # 'sample_step'  QLineEdit for step of colormap sampling
        # 'dark'         QPushButton for toggling dark / standard GUI

        if force_dark:
            self.ui['dark'].setChecked(True)
            self.handle_dark_button(True)
        
        self.open_palette = pg.palette.get('system')
        self.open_palette.apply()
        self.update_color_fields( self.open_palette )

        self.num_points = 30
        
        # configure overview plot with four colors:
        plt = self.ui['plot1']
        plt.enableAutoRange(False)
        plt.setYRange( 0, 4, padding=0 )
        plt.setXRange( 0, self.num_points, padding=0 ) 
        for key in ('left','right','top','bottom'):
            ax = plt.getAxis(key)
            ax.show()
            ax.setZValue(0.1)

        self.curves = []
        curve = pg.PlotCurveItem(pen='p0', brush=('p0',127))
        curve.setFillLevel(0)
        self.curves.append( (1, 1, curve) ) # dataset 1, vertical offset 3        
        plt.addItem(curve)
        curve = pg.ScatterPlotItem(
            symbol='o', size=5, pen='p0', brush='p0', # ('p0',127),
            hoverable=True, hoverPen='gr_acc', hoverBrush='gr_reg')
        self.curves.append( (1, 1, curve) ) # dataset 1, vertical offset 2
        plt.addItem(curve)

        pen_list = ['p2', 'p4', 'p6'] # add three more plots
        for idx, pen in enumerate( pen_list ):
            curve = pg.PlotCurveItem()
            curve.setPen(pen, width=5)
            self.curves.append( (3+2*idx, 1.5+0.8*idx, curve) ) # datasets 2+, vertical offset 3+
            plt.addItem(curve)

        # configure tall plot with eight colors and region overlay:
        plt = self.ui['plot2']
        plt.enableAutoRange(False)
        plt.setYRange( -0.6, 7.6, padding=0 )
        plt.setXRange( 0, self.num_points, padding=0 )
        plt.getAxis('bottom').hide()
        plt.getAxis('left').setLabel('plot color')
        plt.getAxis('left').setGrid(0.5) # 63)

        pen_list = [('p0',2),('p1',2),('p2',2),('p3',2),('p4',2),('p5',2),('p6',2),('p7',2)] # add right-side plots for each main color
        for idx, pen in enumerate( pen_list ):
            curve = pg.PlotCurveItem(pen=pen)
            self.curves.append( (1+idx, idx, curve) ) # datasets 2+, vertical offset by index
            plt.addItem(curve)
        item = pg.LinearRegionItem( values=(4, 8), orientation='vertical' )
        plt.addItem(item)
        
        col_list = [
            ('k','black'),('b','blue'),('c','cyan'),('g','green'),('y','yellow'),
            ('r','red'),('m','magenta'),('w','white')]
        for idx in range(10):
            name = 'm{:d}'.format(idx)
            col_list.append( (name, '## '+name+' ##') )

        for idx, (color,text) in enumerate( col_list ):
            text_item = pg.TextItem(text, color=color, anchor=(1,0))
            text_item.setPos( self.num_points, 7.7 - idx/3 ) 
            plt.addItem(text_item)       

        self.show()

        # prepare for continuous updates and frame rate measurement
        self.last_time = time()
        self.fps = None
        self.timer = QtCore.QTimer(singleShot=False)
        self.timer.timeout.connect( self.timed_update )

        # prepare initial data and display in plots
        self.data = np.zeros((9, self.num_points ))
        self.data[0,:] = np.arange( self.data.shape[1] ) # used as x data
        self.phases = np.zeros(9)
        self.timed_update()


    ### handle GUI interaction ###############################################
    def update_color_fields(self, pal):
        """ update line edit fields for selected palette """
        if pal is None:
            print('palette is None!')
            return
        for key in self.ui['widget_from_color_key']:
            wid = self.ui['widget_from_color_key'][key]
            qcol = pal[key]
            if wid is not None:
                wid.setText( qcol.name() )

    def handle_palette_select(self, idx):
        """ user selected a palette in dropdown menu """
        text, identifier, args = self.palette_options[idx]
        del text # not needed here
        self.open_palette = pg.palette.get(identifier, *args)
        print('loaded palette:', identifier, args)

        if identifier in pg.palette.PALETTE_DEFINITIONS:
            info = pg.palette.PALETTE_DEFINITIONS[identifier]
            colormap_sampling = info['colormap_sampling']
            if colormap_sampling is None:
                identifier, start, step = 'none', 0.000, 0.125
            else:
                identifier, start, step = colormap_sampling
            self.ui['sample_start'].setText('{:+.3f}'.format(start))
            self.ui['sample_step' ].setText('{:+.3f}'.format(step) )
            for idx, map_id in enumerate( self.colormap_options ):
                if map_id == identifier:
                    # print('found colormap at idx',idx)
                    self.ui['colormaps'].setCurrentIndex(idx)
        self.update_color_fields(self.open_palette)
        self.open_palette.apply()
        
    def handle_colormap_select(self, param=None):
        """ user selected a colormap in dropdown menu or changed start / step vales """
        del param # drop index sent by QComboBox
        identifier = self.ui['colormaps'].currentText()
        if identifier == 'none':
            return
        start = self.ui['sample_start'].text()
        step  = self.ui['sample_step' ].text()
        try:
            start = float(start)
        except ValueError:
            start = 0.0
        self.ui['sample_start'].setText('{:+.3f}'.format(start) )
        try:
            step = float(step)
        except ValueError:
            step = 0.125
        self.ui['sample_step'].setText('{:+.3f}'.format(step) )
        self.open_palette.sampleColorMap( cmap=identifier, start=start, step=step )
        # print('applied color map {:s} starting at {:.3f} with step {:3f}'.format(identifier, start, step) )
        
        self.update_color_fields(self.open_palette)
        self.open_palette.apply()
        
    def handle_color_update(self):
        """ figure out what color field was updated """
        source = self.sender()
        key = self.ui['color_key_from_widget'][source]
        requested = source.text()
        if len(requested) < 1:
            value = 0x808080
        else:
            if requested[0] == '#': 
                requested = requested[1:]
            try:
                value = int(requested,16)
            except ValueError:
                value = 0x808080
        color = '#{:06x}'.format(value)
        print('color value is',color)
        # source.setText(color)
        self.open_palette[key] = color
        self.update_color_fields(self.open_palette)
        self.open_palette.apply()
        
        print('color update requested for',key)

    def handle_update_button(self, active):
        """ start/stop timer """
        if active:
            self.timer.start(1)
        else:
            self.timer.stop()

    def handle_dark_button(self, active):
        """ manually switch to dark palette to test on windows """
        app = QtWidgets.QApplication.instance()
        if active:
            app.setPalette( self.q_palette['dark'] ) # apply dark QPalette
        else:
            app.setPalette( self.q_palette['system'] ) # reapply QPalette stored at start-up

    def timed_update(self):
        """ update loop, called by timer """
        self.speed = np.linspace(0.01, 0.06, 9) 
        self.phases += self.speed * np.random.normal(1, 1, size=9)
        for idx in range(1, self.data.shape[0]):
            self.data[idx, :-1] = self.data[idx, 1:] # roll
            # self.data[idx, -1] = np.random.normal()
        self.data[1:, -1] = 0.5 * np.sin( self.phases[1:] )
        xdata = self.data[0,:]
        for idx, offset, curve in self.curves:
            curve.setData( x=xdata, y=( offset + self.data[idx,:] ) )

        now = time()
        dt = now - self.last_time
        self.last_time = now
        if self.fps is None:
            self.fps = 1.0/dt
        else:
            s = np.clip(dt*3., 0, 1)
            self.fps = self.fps * (1-s) + (1.0/dt) * s
        self.ui['plot2'].setTitle('%0.2f fps' % self.fps)
        QtWidgets.QApplication.processEvents()  ## force complete redraw for every plot
        

    ### Qt color definitions for dark palette on Windows #####################
    def make_dark_QPalette(self):
        # color definitions match QDarkstyle
        BLACK      = QtGui.QColor('#000000')
        # BG_LIGHT   = QtGui.QColor('#505F69')
        # BG_NORMAL  = QtGui.QColor('#32414B')
        # BG_DARK    = QtGui.QColor('#19232D')
        BG_LIGHT   = QtGui.QColor('#505354') # #32414B
        BG_NORMAL  = QtGui.QColor('#2e3132') # #212a31
        BG_DARK    = QtGui.QColor('#0e1112') # #10151
        FG_LIGHT   = QtGui.QColor('#f0f4f5') # #F0F0F0
        FG_NORMAL  = QtGui.QColor('#d4d8d9') # #AAAAAA
        FG_DARK    = QtGui.QColor('#b8bcbd') # #787878
        SEL_LIGHT  = QtGui.QColor('#148CD2')
        SEL_NORMAL = QtGui.QColor('#1464A0')
        SEL_DARK   = QtGui.QColor('#14506E')
        qpal = QtGui.QPalette( QtGui.QColor(BG_DARK) )
        for ptype in (  QtGui.QPalette.Active,  QtGui.QPalette.Inactive ):
            qpal.setColor( ptype, QtGui.QPalette.Window, BG_NORMAL )
            qpal.setColor( ptype, QtGui.QPalette.WindowText, FG_LIGHT ) # or white?
            qpal.setColor( ptype, QtGui.QPalette.Base, BG_DARK )
            qpal.setColor( ptype, QtGui.QPalette.Text, FG_LIGHT )
            qpal.setColor( ptype, QtGui.QPalette.AlternateBase, BG_DARK )
            qpal.setColor( ptype, QtGui.QPalette.ToolTipBase, BG_LIGHT )
            qpal.setColor( ptype, QtGui.QPalette.ToolTipText, FG_LIGHT )
            qpal.setColor( ptype, QtGui.QPalette.Button, BG_NORMAL )
            qpal.setColor( ptype, QtGui.QPalette.ButtonText, FG_LIGHT )
            qpal.setColor( ptype, QtGui.QPalette.Link, SEL_NORMAL )
            qpal.setColor( ptype, QtGui.QPalette.LinkVisited, FG_NORMAL )
            qpal.setColor( ptype, QtGui.QPalette.Highlight, SEL_LIGHT )
            qpal.setColor( ptype, QtGui.QPalette.HighlightedText, BLACK )
        qpal.setColor( QtGui.QPalette.Disabled, QtGui.QPalette.Button, BG_NORMAL )
        qpal.setColor( QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, FG_DARK )
        qpal.setColor( QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, FG_DARK )
        return qpal
        
    ##########################################################################
    def prepare_ui(self):
        """ Boring Qt window layout code is implemented here """
        ui = {}
        main_wid = QtWidgets.QWidget()
        self.setCentralWidget(main_wid)
        
        color_fields = (
            # key, description, reference to line edit field)
            ['gr_bg' , (0,0), 'graph background'   ],
            ['gr_fg' , (1,0), 'graph foreground'   ],
            ['gr_txt', (2,0), 'graph text'         ],
            ['gr_reg', (3,0), 'graph region'       ], 
            ['gr_acc', (4,0), 'graphical accent'   ], 
            ['gr_hlt', (5,0), 'graphical highlight'],
            ['p0', (0,1), ' plot 0'], ['p1', (1,1), ' plot 1'],
            ['p2', (2,1), ' plot 2'], ['p3', (3,1), ' plot 3'],
            ['p4', (4,1), ' plot 4'], ['p5', (5,1), ' plot 5'],
            ['p6', (6,1), ' plot 6'], ['p7', (7,1), ' plot 7']            
        )

        gr_wid1 = pg.GraphicsLayoutWidget(show=True)
        ui['plot1'] = gr_wid1.addPlot()

        gr_wid2 = pg.GraphicsLayoutWidget(show=True)
        ui['plot2'] = gr_wid2.addPlot()

        main_layout = QtWidgets.QHBoxLayout( main_wid )
        l_wid = QtWidgets.QWidget()
        main_layout.addWidget(l_wid)
        main_layout.addWidget(gr_wid2)
        
        l_layout = QtWidgets.QGridLayout( l_wid )
        l_layout.setContentsMargins(0,0,0,0)
        l_layout.setSpacing(1)
        row_idx = 0

        label = QtWidgets.QLabel('System style:')
        l_layout.addWidget( label, row_idx,0, 1,2 )

        label = QtWidgets.QLabel('Select a palette:')
        l_layout.addWidget( label, row_idx,2, 1,2 )
        row_idx += 1

        btn = QtWidgets.QPushButton('dark GUI')
        btn.setCheckable(True)
        btn.setChecked(False)
        btn.clicked.connect(self.handle_dark_button)

        box = QtWidgets.QComboBox()
        for text, identifier, args in self.palette_options:
            del identifier, args # not needed here
            box.addItem(text)
        box.activated.connect(self.handle_palette_select)

        l_layout.addWidget( box, row_idx,2, 1,2 )
        l_layout.addWidget( btn, row_idx,0, 1,2 )
        ui['dark'] = btn
        row_idx += 1

        label = QtWidgets.QLabel('Sampled color map:')
        l_layout.addWidget( label, row_idx,0, 1,2 )
        label = QtWidgets.QLabel('start')
        l_layout.addWidget( label, row_idx,2, 1,1 )
        label = QtWidgets.QLabel('step')
        l_layout.addWidget( label, row_idx,3, 1,1 )
        row_idx += 1
        
        box = QtWidgets.QComboBox()
        for identifier in self.colormap_options:
            box.addItem(identifier)
        ui['colormaps'] = box
        ui['colormaps'].activated.connect(self.handle_colormap_select)
        ui['sample_start'] = QtWidgets.QLineEdit(' 0.000')
        ui['sample_start'].editingFinished.connect(self.handle_colormap_select)
        ui['sample_step' ] = QtWidgets.QLineEdit('+0.125')
        ui['sample_step' ].editingFinished.connect(self.handle_colormap_select)
        l_layout.addWidget( box, row_idx,0, 1,2 )
        l_layout.addWidget( ui['sample_start'], row_idx,2, 1,1 )
        l_layout.addWidget( ui['sample_step' ], row_idx,3, 1,1 )
        row_idx += 1

        spacer = QtWidgets.QWidget()
        spacer.setFixedHeight(10)
        l_layout.addWidget( spacer, row_idx,0, 1,2 )
        row_idx += 1

        label = QtWidgets.QLabel('Functional colors:')
        l_layout.addWidget( label, row_idx,0, 1,2 )
        row_idx += 1

        row =  0
        ui['widget_from_color_key'] = {} # look-up for color editing fields
        ui['color_key_from_widget'] = {} # reverse look-up for color editing fields
        for field_list in color_fields:
            key, pos, text = field_list
            lab  = QtWidgets.QLabel(text)
            lab.setAlignment(QtCore.Qt.AlignCenter)
            edt = QtWidgets.QLineEdit()
            edt.editingFinished.connect(self.handle_color_update)
            row = row_idx + pos[0]
            col = 2 * pos[1]  # 0 or 2
            l_layout.addWidget( lab, row,col+0, 1,1 )
            l_layout.addWidget( edt, row,col+1, 1,1 )
            ui['color_key_from_widget'][edt] = key
            ui['widget_from_color_key'][key] = edt
        row_idx = row
            
        btn = QtWidgets.QPushButton('generate continuous data')
        btn.setCheckable(True)
        btn.setChecked(False)
        btn.clicked.connect(self.handle_update_button)
        l_layout.addWidget( btn, row_idx,0, 1,2 )
        row_idx += 1

        spacer = QtWidgets.QWidget()
        spacer.setFixedHeight(10)
        l_layout.addWidget( spacer, row_idx,0, 1,2 )
        row_idx += 1

        label = QtWidgets.QLabel('Overview:')
        l_layout.addWidget( label, row_idx,0, 1,4 )
        row_idx += 1
        
        l_layout.addWidget( gr_wid1, row_idx,0, 1,4 )
        row_idx += 1

        return ui



mkQApp("Palette test application")
main_window = MainWindow()

## Start Qt event loop
if __name__ == '__main__':
    QtWidgets.QApplication.instance().exec_()
