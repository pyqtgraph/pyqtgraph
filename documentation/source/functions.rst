Pyqtgraph's Helper Functions
============================

Simple Data Display Functions
-----------------------------

.. autofunction:: pyqtgraph.plot

.. autofunction:: pyqtgraph.image



Color, Pen, and Brush Functions
-------------------------------

Qt uses the classes QColor, QPen, and QBrush to determine how to draw lines and fill shapes. These classes are highly capable but somewhat awkward to use. Pyqtgraph offers the functions :func:`~pyqtgraph.mkColor`, :func:`~pyqtgraph.mkPen`, and :func:`~pyqtgraph.mkBrush` to simplify the process of creating these classes. In most cases, however, it will be unnecessary to call these functions directly--any function or method that accepts *pen* or *brush* arguments will make use of these functions for you. For example, the following three lines all have the same effect::
    
    pg.plot(xdata, ydata, pen='r')
    pg.plot(xdata, ydata, pen=pg.mkPen('r'))
    pg.plot(xdata, ydata, pen=QPen(QColor(255, 0, 0)))


.. autofunction:: pyqtgraph.mkColor

.. autofunction:: pyqtgraph.mkPen

.. autofunction:: pyqtgraph.mkBrush

.. autofunction:: pyqtgraph.hsvColor

.. autofunction:: pyqtgraph.intColor

.. autofunction:: pyqtgraph.colorTuple

.. autofunction:: pyqtgraph.colorStr


Data Slicing
------------

.. autofunction:: pyqtgraph.affineSlice



SI Unit Conversion Functions
----------------------------

.. autofunction:: pyqtgraph.siFormat

.. autofunction:: pyqtgraph.siScale

.. autofunction:: pyqtgraph.siEval


Image Preparation Functions
---------------------------

.. autofunction:: pyqtgraph.makeARGB

.. autofunction:: pyqtgraph.makeQImage


Mesh Generation Functions
-------------------------

.. autofunction:: pyqtgraph.isocurve

.. autofunction:: pyqtgraph.isosurface



