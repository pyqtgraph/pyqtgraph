[![Build Status](https://travis-ci.org/pyqtgraph/pyqtgraph.svg?branch=develop)](https://travis-ci.org/pyqtgraph/pyqtgraph)
[![codecov.io](http://codecov.io/github/pyqtgraph/pyqtgraph/coverage.svg?branch=develop)](http://codecov.io/github/pyqtgraph/pyqtgraph?branch=develop)

PyQtGraph
=========

A pure-Python graphics library for PyQt/PySide

Copyright 2017 Luke Campagnola, University of North Carolina at Chapel Hill

<http://www.pyqtgraph.org>

PyQtGraph is intended for use in mathematics / scientific / engineering applications.
Despite being written entirely in python, the library is fast due to its
heavy leverage of numpy for number crunching, Qt's GraphicsView framework for
2D display, and OpenGL for 3D display.


Requirements
------------

  * PyQt 4.7+, PySide, or PyQt5
  * python 2.7, or 3.x
  * NumPy
  * For 3D graphics: pyopengl and qt-opengl
  * Known to run on Windows, Linux, and Mac.

Support
-------
  
  * Report issues on the [GitHub issue tracker](https://github.com/pyqtgraph/pyqtgraph/issues)
  * Post questions to the [mailing list / forum](https://groups.google.com/forum/?fromgroups#!forum/pyqtgraph) or [StackOverflow](https://stackoverflow.com/questions/tagged/pyqtgraph)

Installation Methods
--------------------

  * From pypi:  
        `pip install pyqtgraph`
  * To use with a specific project, simply copy the pyqtgraph subdirectory
    anywhere that is importable from your project. 
  * To install system-wide from source distribution:
        `$ python setup.py install`
  * For installation packages, see the website (pyqtgraph.org)

Documentation
-------------

The easiest way to learn pyqtgraph is to browse through the examples; run `python -m pyqtgraph.examples` for a menu.

The official documentation lives at http://pyqtgraph.org/documentation

