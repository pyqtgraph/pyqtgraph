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
* (index, maximum) tuple for automatically iterating through colors (see :func:`~pyqtgraph.intColor`)
* :qt:`QColor`
* :qt:`QPen` / :qt:`QBrush` where appropriate

Notably, more complex pens and brushes can be easily built using the 
:func:`~pyqtgraph.mkPen` and :func:`~pyqtgraph.mkBrush` functions or with :term:`Qt`'s :qt:`QPen` and :qt:`QBrush` classes::

    mkPen('y', width=3, style=QtCore.Qt.DashLine)          # A dashed yellow line 2px wide
    mkPen(0.5)                                             # Solid grey line 1px wide
    mkPen(color=(200, 200, 255), style=QtCore.Qt.DotLine)  # Dotted pale-blue line
    
See the Qt documentation for :qt:`QPen` and 'PenStyle' for more line-style options and :qt:`QBrush` for more fill options.
Colors can also be built using :func:`~pyqtgraph.mkColor`, 
:func:`~pyqtgraph.intColor`, :func:`~pyqtgraph.hsvColor`, or Qt's :qt:`QColor` class.


Default Background and Foreground Colors
----------------------------------------

By default, PyQtGraph uses a black background for its plots and grey for axes, text, and plot lines.
These defaults can be changed using :func:`~pyqtgraph.setConfigOption`::
    
    import pyqtgraph as pg

    ## Switch to using white background and black foreground
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')

    ## The following plot has inverted colors
    pg.plot([1,4,2,3,5])
  
.. warning::
    This must be set *before* creating any widgets


