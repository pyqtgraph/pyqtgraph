Introduction
============



What is pyqtgraph?
------------------

PyQtGraph is a graphics and user interface library for Python that provides
functionality commonly required in engineering and science applications. Its
primary goals are 1) to provide fast, interactive graphics for displaying data
(plots, video, etc.) and 2) to provide tools to aid in rapid application
development (for example, property trees such as used in Qt Designer).

PyQtGraph makes heavy use of the Qt GUI platform (via PyQt or PySide) for its
high-performance graphics and numpy for heavy number crunching. In particular,
pyqtgraph uses Qt's GraphicsView framework which is a highly capable graphics
system on its own; we bring optimized and simplified primitives to this
framework to allow data visualization with minimal effort.

It is known to run on Linux, Windows, and OSX


What can it do?
---------------

Amongst the core features of pyqtgraph are:

* Basic data visualization primitives: Images, line and scatter plots
* Fast enough for realtime update of video/plot data
* Interactive scaling/panning, averaging, FFTs, SVG/PNG export
* Widgets for marking/selecting plot regions
* Widgets for marking/selecting image region-of-interest and automatically
  slicing multi-dimensional image data
* Framework for building customized image region-of-interest widgets
* Docking system that replaces/complements Qt's dock system to allow more
  complex (and more predictable) docking arrangements
* ParameterTree widget for rapid prototyping of dynamic interfaces (Similar to
  the property trees in Qt Designer and many other applications)


.. _examples:

Examples
--------

PyQtGraph includes an extensive set of examples that can be accessed by
running::

    import pyqtgraph.examples
    pyqtgraph.examples.run()

Or by running ``python examples/`` from the source root.

This will start a launcher with a list of available examples. Select an item
from the list to view its source code and double-click an item to run the
example.

Note If you have installed pyqtgraph with ``python setup.py develop``
then the examples are incorrectly exposed as a top-level module. In this case,
use ``import examples; examples.run()``.


How does it compare to...
-------------------------

* matplotlib: For plotting, pyqtgraph is not nearly as complete/mature as
  matplotlib, but runs much faster. Matplotlib is more aimed toward making
  publication-quality graphics, whereas pyqtgraph is intended for use in data
  acquisition and analysis applications. Matplotlib is more intuitive for
  matlab programmers; pyqtgraph is more intuitive for python/qt programmers.
  Matplotlib (to my knowledge) does not include many of pyqtgraph's features
  such as image interaction, volumetric rendering, parameter trees,
  flowcharts, etc.

* pyqwt5: About as fast as pyqwt5, but not quite as complete for plotting
  functionality. Image handling in pyqtgraph is much more complete (again, no
  ROI widgets in qwt). Also, pyqtgraph is written in pure python, so it is
  more portable than pyqwt, which often lags behind pyqt in development (I
  originally used pyqwt, but decided it was too much trouble to rely on it
  as a dependency in my projects). Like matplotlib, pyqwt (to my knowledge)
  does not include many of pyqtgraph's features such as image interaction,
  volumetric rendering, parameter trees, flowcharts, etc.

(My experience with these libraries is somewhat outdated; please correct me if
I am wrong here)
