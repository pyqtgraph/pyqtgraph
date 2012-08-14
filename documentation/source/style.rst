Line, Fill, and Color
=====================

Qt relies on its QColor, QPen and QBrush classes for specifying line and fill styles for all of its drawing.
Internally, pyqtgraph uses the same system but also allows many shorthand methods of specifying
the same style options.

Many functions and methods in pyqtgraph accept arguments specifying the line style (pen), fill style (brush), or color. 
For most of these function arguments, the following values may be used:
    
* single-character string representing color (b, g, r, c, m, y, k, w)
* (r, g, b) or (r, g, b, a) tuple
* single greyscale value (0.0 - 1.0)
* (index, maximum) tuple for automatically iterating through colors (see :func:`intColor <pyqtgraph.intColor>`)
* QColor
* QPen / QBrush where appropriate

Notably, more complex pens and brushes can be easily built using the 
:func:`mkPen() <pyqtgraph.mkPen>` / :func:`mkBrush() <pyqtgraph.mkBrush>` functions or with Qt's QPen and QBrush classes::

    mkPen('y', width=3, style=QtCore.Qt.DashLine)          ## Make a dashed yellow line 2px wide
    mkPen(0.5)                                             ## solid grey line 1px wide
    mkPen(color=(200, 200, 255), style=QtCore.Qt.DotLine)  ## Dotted pale-blue line
    
See the Qt documentation for 'QPen' and 'PenStyle' for more line-style options and 'QBrush' for more fill options.
Colors can also be built using :func:`mkColor() <pyqtgraph.mkColor>`, 
:func:`intColor() <pyqtgraph.intColor>`, :func:`hsvColor() <pyqtgraph.hsvColor>`, or Qt's QColor class.


Default Background and Foreground Colors
----------------------------------------

By default, pyqtgraph uses a black background for its plots and grey for axes, text, and plot lines.
These defaults can be changed using pyqtgraph.setConfigOption()::
    
    import pyqtgraph as pg

    ## Switch to using white background and black foreground
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')

    ## The following plot has inverted colors
    pg.plot([1,4,2,3,5])
    
(Note that this must be set *before* creating any widgets)


