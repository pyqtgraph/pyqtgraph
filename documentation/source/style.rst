Line, Fill, and Color
=====================

Many functions and methods in pyqtgraph accept arguments specifying the line style (pen), fill style (brush), or color. 

For these function arguments, the following values may be used:
    
* single-character string representing color (b, g, r, c, m, y, k, w)
* (r, g, b) or (r, g, b, a) tuple
* single greyscale value (0.0 - 1.0)
* (index, maximum) tuple for automatically iterating through colors (see functions.intColor)
* QColor
* QPen / QBrush where appropriate

Notably, more complex pens and brushes can be easily built using the mkPen() / mkBrush() functions or with Qt's QPen and QBrush classes::

    mkPen('y', width=3, style=QtCore.Qt.DashLine)          ## Make a dashed yellow line 2px wide
    mkPen(0.5)                                             ## solid grey line 1px wide
    mkPen(color=(200, 200, 255), style=QtCore.Qt.DotLine)  ## Dotted pale-blue line
    
See the Qt documentation for 'QPen' and 'PenStyle' for more options.
Colors can also be built using mkColor(), intColor(), hsvColor(), or Qt's QColor class
