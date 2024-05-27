.. role:: python(code)
   :language: python

ImageItem
=========

Overview
--------

:class:`~pyqtgraph.ImageItem` displays images inside a
:class:`~pyqtgraph.GraphicsView`, or a :class:`~pyqtgraph.ViewBox`, which may itself be
part of a :class:`~pyqtgraph.PlotItem`. It is designed for rapid updates as needed for
a video display. The supplied data is optionally scaled (see 
:meth:`ImageItem.setLevels <pyqtgraph.ImageItem.setLevels>`) and/or colored according
to a lookup table (see :meth:`ImageItem.setColorMap <pyqtgraph.ImageItem.setColorMap>`
and  :meth:`ImageItem.setLookupTable <pyqtgraph.ImageItem.setLookupTable>`).

Data is provided as a NumPy array with an ordering of either

* `col-major`, where the shape of the array represents (width, height) or
* `row-major`, where the shape of the array represents (height, width).

While `col-major` is the default, `row-major` ordering typically has the best
performance. In either ordering, a third dimension can be added to the array to hold
individual :python:`[R,G,B]` or :python:`[R,G,B,A]` channels/components.

Notes
-----

:class:`~pyqtgraph.ImageItem` is frequently used in conjunction with
:class:`~pyqtgraph.ColorBarItem` to provide a color map display and interactive level
adjustments, or with :class:`~pyqtgraph.HistogramLUTItem` or
:class:`~pyqtgraph.HistogramLUTWidget` for a full GUI to control the levels and lookup
table used to display the image.

An image can be placed into a plot area of a given extent directly through the
:meth:`ImageItem.setRect <pyqtgraph.ImageItem.setRect>` method or the `rect` keyword.
This is internally realized through assigning a :class:`QTransform`. For other
translation, scaling or rotations effects that persist for all later image data, the
user can also directly define and assign such a transform, as shown in the example
below.

.. _ImageItem_performance:

Performance
-----------

The performance of :class:`~pyqtgraph.ImageItem` can vary *significantly* based on
attributes of the `image`, `levels` and `lut` input arguments. It should not be
assumed that the default parameters are the most performant, as the default values are 
largely there to preserve backwards compatibility.

The following guidance should be observed if performance is an important factor

* Instantiate :class:`~pyqtgraph.ImageItem` with :python:`axisOrder='row-major'`
  
  * Alternatively, set the global configuration optionally
    :python:`pyqtgraph.setConfigOption('imageAxisOrder', 'row-major')`

* Use C-contiguous image data.
* For 1 or 3 channel data, use `uint8`, `uint16`, `float32`, or `float64` `image`
  dtype.
* For 4-channel data, use `uint8` or `uint16` with :python:`levels=None`.
* `levels` should be set to either to ``None`` or to single channel ``[min, max]``
  
  * Not setting `levels` will trigger autoLevels sampling 

* If using LUTs (lookup tables), ensure they have a dtype of `uint8` and have 256
  points or less. That can be accomplished with calling:

  * :func:`ImageItem.setColorMap <pyqtgraph.ImageItem.setColorMap>` or
  * :func:`ImageItem.setLookupTable <pyqtgraph.ImageItem.setLookupTable>` with 
    :python:`ColorMap.getLookupTable(nPts=256)` (default is :python:`nPts=512`)

* For floating point `image` arrays, prefer `float32` dtype to `float64` and avoid
  ``NaN`` values.
* Enable Numba with :python:`pyqtgraph.setConfigOption('useNumba', True)`

  * JIT compilation will only accelerate repeated image display.

Internally, pyqtgraph attempts to directly construct a :class:`QImage` using a
combination of :class:`QImage.Format <QImage.Format>` options and
:meth:`QImage.setColorTable <QImage.setColorTable>` if necessary. This does not work in
all cases that pyqtgraph supports.  If pyqtgraph is unable to construct the
:class:`QImage` in such a fashion, it will fall back on
:func:`~pyqtgraph.functions.makeARGB` to manipulate the data in a manner that
:class:`QImage` can read it in.  There is a *significant* performance penalty when
having to use :func:`~pyqtgraph.functions.makeARGB`.

For applications that are *very* performance sensitive, every effort should be made so
that the arguments passed to :meth:`ImageItem.setImage <pyqtgraph.ImageItem.setImage>`
do not call :func:`~pyqtgraph.functions.makeARGB`.

.. _ImageItem_examples:

Examples
--------

Scale and Position ImageItem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the following example, it is demonstrated how a user scale and translate a
:class:`~pyqtgraph.ImageItem` within a :class:`~pyqtgraph.ViewBox` to occupy a specific
position and size.

.. literalinclude:: /images/gen_example_imageitem_transform.py
    :lines: 4-44
    :emphasize-lines: 24-33
    :language: python

.. thumbnail:: 
    /images/example_imageitem_transform.png
    :width: 49%
    :alt: Example of transformed image display
    :title: Transformed Image Display

Inheritance
-----------

.. inheritance-diagram:: pyqtgraph.graphicsItems.ImageItem.ImageItem
  :top-classes: PyQt6.QtCore.QObject, PyQt6.QtWidgets.QGraphicsItem
  :parts: 1

API
---

.. autoclass:: pyqtgraph.ImageItem
    :members:
