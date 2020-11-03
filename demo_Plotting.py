# -*- coding: utf-8 -*-
"""
This example demonstrates many of the 2D plotting capabilities
in pyqtgraph. All of the plots may be panned/scaled by dragging with 
the left/right mouse buttons. Right click on any plot to show a context menu.
Now with added color palette management (and swapping) through ColorMap object.
"""
# import initExample ## Add path to library (just for examples; you do not need this)
import numpy as np

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

#QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])

win = pg.GraphicsLayoutWidget() # title="Basic plotting examples")
win.show()
# win.setBackgroundColor('w')
win.resize(1000,600)
win.setWindowTitle('pyqtgraph example: Plotting')

#------------------------------------------------------------------------------
# pre-load color maps/palettes
cmap_l = pg.colormap.get('material_style_light')
cmap_l.names['fg'] = cmap_l.names['grey-900']
cmap_l.names['bg'] = cmap_l.names['grey-50']

cmap_d = pg.colormap.get('material_style_dark')
cmap_d.names['fg'] = cmap_d.names['grey-200']
cmap_d.names['bg'] = cmap_d.names['grey-900']

cmap_cet = pg.colormap.get('CET-C1s.csv')
cmap = cmap_d

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

plot_list = []
item_color_list = []

def set_colors(cmap):
    win.setBackground( cmap['bg'])
    for (plot, title) in plot_list:
        fg_col = cmap['fg']
        fg_hex = fg_col.name()
        plot.setTitle(title, **{'color': fg_hex})  # 'color': '#FFF'
        for loc in ['left','right','top','bottom']:
            ax = plot.getAxis(loc)
            ax.setPen(fg_col)
            ax.setTextPen(fg_col)
    for (item, part, col_name) in item_color_list:
        if part == 'pen':
            item.setPen( cmap[col_name] )
        elif part == 'symbolPen':
            item.setSymbolPen( cmap[col_name] )
        elif part == 'symbolBrush':
            item.setSymbolBrush( cmap[col_name] )
        elif part == 'symbolBrushAlpha':
            col = cmap[col_name]
            col.setAlpha(50)
            item.setSymbolBrush( col )
        
#------------------------------------------------------------------------------
# demonstrate shorthand access to named colors in colormap palette
p2 = win.addPlot(title="Multiple curves")
item1 = p2.plot(np.random.normal(size=100), pen='r', name="Red curve")
item2 = p2.plot(np.random.normal(size=110)+5, pen='r', name="Green curve")
item3 = p2.plot(np.random.normal(size=120)+10, pen='r', name="Blue curve")

plot_list.append( (p2, 'Multiple curves'))
item_color_list.append( (item1, 'pen', 'red') )
item_color_list.append( (item2, 'pen', 'green') )
item_color_list.append( (item3, 'pen', 'blue') )

#------------------------------------------------------------------------------
# demonstrate indexed access to colors in colormap palette
p3 = win.addPlot(title="Drawing with points")
item1 = p3.plot(np.random.normal(size=100), pen=cmap['orange'], symbolBrush=cmap['orange'], symbolPen=cmap['fg'])

plot_list.append( (p3, 'Drawing with points'))
item_color_list.append( (item1, 'pen', 'orange') )
item_color_list.append( (item1, 'symbolBrush', 'orange') )
item_color_list.append( (item1, 'symbolPen', 'fg') )

win.nextRow()
#------------------------------------------------------------------------------
# demonstrate slightly awkward adjustement of color parameters
p5 = win.addPlot(title="Scatter plot, axis labels, log scale")
x = np.random.normal(size=1000) * 1e-5
y = x*1000 + 0.005 * np.random.normal(size=1000)
y -= y.min()-1.0
mask = x > 1e-15
x = x[mask]
y = y[mask]
item1 = p5.plot(x, y, pen=None, symbol='t', symbolPen=None, symbolSize=10, symbolBrush='k')
p5.setLabel('left', "Y Axis", units='A')
p5.setLabel('bottom', "Y Axis", units='s')
p5.setLogMode(x=True, y=False)

plot_list.append( (p5, "Scatter plot, axis labels, log scale" ))
item_color_list.append( (item1, 'symbolBrushAlpha', 'deep-purple') )

#------------------------------------------------------------------------------
# demonstrate color mapped access to colors in map
p6 = win.addPlot(title="Updating plot")
plot_list.append( (p6, 'Updating plot'))

curve = p6.plot(pen='y')
data = np.random.normal(size=(10,1000))
ptr = 0

in_dark_mode = True
set_colors(cmap_d)
def update():
    global curve, data, ptr, p6, in_dark_mode
    curve.setData(data[ptr%10])
    col_map = ptr / 200.002
    col_map = col_map - int(col_map) # keep to 0.0 to 1.0
    curve.setPen( cmap_cet[col_map] )
    if ptr == 0:
        p6.enableAutoRange('xy', False)  ## stop auto-scaling after the first data set is plotted
    if ptr % 50 == 0:
        if in_dark_mode:
            in_dark_mode = False
            cmap = cmap_l
        else:
            in_dark_mode = True
            cmap = cmap_d            
        set_colors(cmap)
    ptr += 1

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
