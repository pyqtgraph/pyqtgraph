PyQtGraph
=========

[![PyPi](https://img.shields.io/pypi/v/pyqtgraph.svg)](https://pypi.org/project/pyqtgraph/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/pyqtgraph.svg)](https://anaconda.org/conda-forge/pyqtgraph)
[![Build Status](https://github.com/pyqtgraph/pyqtgraph/workflows/main/badge.svg)](https://github.com/pyqtgraph/pyqtgraph/actions/?query=workflow%3Amain)
[![CodeQL Status](https://github.com/pyqtgraph/pyqtgraph/workflows/codeql/badge.svg)](https://github.com/pyqtgraph/pyqtgraph/actions/?query=workflow%3Acodeql)
[![Documentation Status](https://readthedocs.org/projects/pyqtgraph/badge/?version=latest)](https://pyqtgraph.readthedocs.io/en/latest/?badge=latest)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/pyqtgraph/pyqtgraph.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/pyqtgraph/pyqtgraph/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/pyqtgraph/pyqtgraph.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/pyqtgraph/pyqtgraph/context:python)

A pure-Python graphics library for PyQt5/PySide2

Copyright 2020 Luke Campagnola, University of North Carolina at Chapel Hill

<http://www.pyqtgraph.org>

PyQtGraph is intended for use in mathematics / scientific / engineering applications.
Despite being written entirely in python, the library is fast due to its
heavy leverage of numpy for number crunching, Qt's GraphicsView framework for
2D display, and OpenGL for 3D display.

Requirements
------------

pyqtgraph has adopted [NEP 29](https://numpy.org/neps/nep-0029-deprecation_policy.html).

This project supports:

* All minor versions of Python released 42 months prior to the project, and at minimum the two latest minor versions.
* All minor versions of numpy released in the 24 months prior to the project, and at minimum the last three minor versions.
* All minor versions of Qt 5 currently supported by upstream Qt (Note, Qt 6 support is not yet implemented)

Currently this means:

* Python 3.7+
* Qt 5.12-5.15
* Required
  * PyQt5 or PySide2
  * `numpy` 1.17+
* Optional
  * `scipy` for image processing
  * `pyopengl` for 3D graphics
    * `pyopengl` on macOS Big Sur only works with python 3.9.1+
  * `hdf5` for large hdf5 binary format support
  * `colorcet` for supplimental colormaps
  * [`cupy`](https://docs.cupy.dev/en/stable/install.html) for CUDA-enhanced image processing
    * On Windows, CUDA toolkit must be >= 11.1

Qt Bindings Test Matrix
-----------------------

The following table represents the python environments we test in our CI system.  Our CI system uses Ubuntu 18.04, Windows Server 2019, and macOS 10.15 base images.

| Qt-Bindings    | Python 3.7         | Python 3.8         | Python 3.9         |
| :------------- | :----------------: | :----------------: | :----------------: |
| PySide2-5.12   | :white_check_mark: | :x:                | :x:                |
| PyQt5-5.12     | :white_check_mark: | :x:                | :x:                |
| PySide2-5.15   | :x:                | :white_check_mark: | :white_check_mark: |
| PyQt5-5.15     | :x:                | :white_check_mark: | :white_check_mark: |

Support
-------
  
* Report issues on the [GitHub issue tracker](https://github.com/pyqtgraph/pyqtgraph/issues)
* Post questions to the [mailing list / forum](https://groups.google.com/forum/?fromgroups#!forum/pyqtgraph) or [StackOverflow](https://stackoverflow.com/questions/tagged/pyqtgraph)

Installation Methods
--------------------

* From PyPI:  
  * Last released version: `pip install pyqtgraph`
  * Latest development version: `pip install git+https://github.com/pyqtgraph/pyqtgraph@master`
* From conda
  * Last released version: `conda install -c conda-forge pyqtgraph`
* To install system-wide from source distribution: `python setup.py install`
* Many linux package repositories have release versions.
* To use with a specific project, simply copy the pyqtgraph subdirectory
  anywhere that is importable from your project.

Documentation
-------------

The official documentation lives at [pyqtgraph.readthedocs.io](https://pyqtgraph.readthedocs.io)

The easiest way to learn pyqtgraph is to browse through the examples; run `python -m pyqtgraph.examples` to launch the examples application.  
