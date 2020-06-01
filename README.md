
[![Build Status](https://pyqtgraph.visualstudio.com/pyqtgraph/_apis/build/status/pyqtgraph.pyqtgraph?branchName=develop)](https://pyqtgraph.visualstudio.com/pyqtgraph/_build/latest?definitionId=17&branchName=develop)
[![Documentation Status](https://readthedocs.org/projects/pyqtgraph/badge/?version=latest)](https://pyqtgraph.readthedocs.io/en/latest/?badge=latest)

PyQtGraph
=========

A pure-Python graphics library for PyQt/PySide/PyQt5/PySide2

Copyright 2019 Luke Campagnola, University of North Carolina at Chapel Hill

<http://www.pyqtgraph.org>

PyQtGraph is intended for use in mathematics / scientific / engineering applications.
Despite being written entirely in python, the library is fast due to its
heavy leverage of numpy for number crunching, Qt's GraphicsView framework for
2D display, and OpenGL for 3D display.

Requirements
------------

* PyQt 4.8+, PySide, PyQt5, or PySide2
  * PySide2 5.14 does not have loadUiType functionality, and thus the example application will not work.  You can follow along with restoring that functionality [here](https://bugreports.qt.io/browse/PYSIDE-1223).
* Python 2.7, or 3.x
* Required
  * `numpy`
* Optional
  * `scipy` for image processing
  * `pyopengl` for 3D graphics
    * macOS with Python2 and Qt4 bindings (PyQt4 or PySide) do not work with 3D OpenGL graphics
    * `pyqtgraph.opengl` will be depreciated in a future version and replaced with `VisPy`
  * `hdf5` for large hdf5 binary format support
* Known to run on Windows, Linux, and macOS.

Qt Bindings Test Matrix
-----------------------

The following table represents the python environments we test in our CI system.  Our CI system uses Ubuntu 18.04, Windows Server 2019, and macOS 10.15 base images.

| Qt-Bindings  | Python 2.7         | Python 3.6         | Python 3.7         | Python 3.8         |
| :----------- | :----------------: | :----------------: | :----------------: | :----------------: |
| PyQt-4       | :white_check_mark: | :x:                | :x:                | :x:                |
| PySide1      | :white_check_mark: | :x:                | :x:                | :x:                |
| PyQt-5.9     | :x:                | :white_check_mark: | :x:                | :x:                |
| PySide2-5.13 | :x:                | :x:                | :white_check_mark: | :x:                |
| PyQt-5.14    | :x:                | :x:                | :x:                | :white_check_mark: |

* pyqtgraph has had some incompatabilities with PySide2-5.6, and we recommend you avoid those bindings if possible
* on macOS with Python 2.7 and Qt4 bindings (PyQt4 or PySide) the openGL related visualizations do not work

Support
-------
  
* Report issues on the [GitHub issue tracker](https://github.com/pyqtgraph/pyqtgraph/issues)
* Post questions to the [mailing list / forum](https://groups.google.com/forum/?fromgroups#!forum/pyqtgraph) or [StackOverflow](https://stackoverflow.com/questions/tagged/pyqtgraph)

Installation Methods
--------------------

* From PyPI:  
  * Last released version: `pip install pyqtgraph`
  * Latest development version: `pip install git+https://github.com/pyqtgraph/pyqtgraph@develop`
* From conda
  * Last released version: `conda install pyqtgraph`
* To install system-wide from source distribution: `python setup.py install`
* Many linux package repositories have release versions.
* To use with a specific project, simply copy the pyqtgraph subdirectory
  anywhere that is importable from your project.
* For installation packages, see the website (pyqtgraph.org)

Documentation
-------------

The easiest way to learn pyqtgraph is to browse through the examples; run `python -m pyqtgraph.examples` for a menu.

The official documentation lives at https://pyqtgraph.readthedocs.io
