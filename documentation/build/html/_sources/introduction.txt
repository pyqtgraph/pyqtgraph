Introduction
============



What is pyqtgraph?
------------------

Pyqtgraph is a graphics and user interface library for Python that provides functionality commonly required in engineering and science applications. Its primary goals are 1) to provide fast, interactive graphics for displaying data (plots, video, etc.) and 2) to provide tools to aid in rapid application development (for example, property trees such as used in Qt Designer).

Pyqtgraph makes heavy use of the Qt GUI platform (via PyQt or PySide) for its high-performance graphics and numpy for heavy number crunching. In particular, pyqtgraph uses Qt's GraphicsView framework which is a highly capable graphics system on its own; we bring optimized and simplified primitives to this framework to allow data visualization with minimal effort. 

It is known to run on Linux, Windows, and OSX


What can it do?
---------------

Amongst the core features of pyqtgraph are:

* Basic data visualization primitives: Images, line and scatter plots
* Fast enough for realtime update of video/plot data
* Interactive scaling/panning, averaging, FFTs, SVG/PNG export
* Widgets for marking/selecting plot regions
* Widgets for marking/selecting image region-of-interest and automatically slicing multi-dimensional image data
* Framework for building customized image region-of-interest widgets
* Docking system that replaces/complements Qt's dock system to allow more complex (and more predictable) docking arrangements
* ParameterTree widget for rapid prototyping of dynamic interfaces (Similar to the property trees in Qt Designer and many other applications)


.. _examples:

Examples
--------

Pyqtgraph includes an extensive set of examples that can be accessed by running::
    
    import pyqtgraph.examples
    pyqtgraph.examples.run()

This will start a launcher with a list of available examples. Select an item from the list to view its source code and double-click an item to run the example.


How does it compare to...
-------------------------

* matplotlib: For plotting and making publication-quality graphics, matplotlib is far more mature than pyqtgraph. However, matplotlib is also much slower and not suitable for applications requiring realtime update of plots/video or rapid interactivity. It also does not provide any of the GUI tools and image interaction/slicing functionality in pyqtgraph.

* pyqwt5: pyqwt is generally more mature than pyqtgraph for plotting and is about as fast. The major differences are 1) pyqtgraph is written in pure python, so it is somewhat more portable than pyqwt, which often lags behind pyqt in development (and can be a pain to install on some platforms) and 2) like matplotlib, pyqwt does not provide any of the GUI tools and image interaction/slicing functionality in pyqtgraph.

(My experience with these libraries is somewhat outdated; please correct me if I am wrong here)
