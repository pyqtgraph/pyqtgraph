PyQtGraph
=========

[![PyPi](https://img.shields.io/pypi/v/pyqtgraph.svg)](https://pypi.org/project/pyqtgraph/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/pyqtgraph.svg)](https://anaconda.org/conda-forge/pyqtgraph)
[![Build Status](https://github.com/pyqtgraph/pyqtgraph/workflows/main/badge.svg)](https://github.com/pyqtgraph/pyqtgraph/actions/?query=workflow%3Amain)
[![CodeQL Status](https://github.com/pyqtgraph/pyqtgraph/workflows/codeql/badge.svg)](https://github.com/pyqtgraph/pyqtgraph/actions/?query=workflow%3Acodeql)
[![Documentation Status](https://readthedocs.org/projects/pyqtgraph/badge/?version=latest)](https://pyqtgraph.readthedocs.io/en/latest/?badge=latest)
[![Discord](https://img.shields.io/discord/946624673200893953.svg?label=PyQtGraph&logo=discord)](https://discord.gg/ufTVNNreAZ)
A pure-Python graphics library for PyQt5/PyQt6/PySide2/PySide6

Copyright 2023 PyQtGraph developers

<http://www.pyqtgraph.org>

PyQtGraph is intended for use in mathematics / scientific / engineering applications.
Despite being written entirely in python, the library is fast due to its
heavy leverage of numpy for number crunching, Qt's GraphicsView framework for
2D display, and OpenGL for 3D display.

Requirements
------------

PyQtGraph has adopted [NEP 29](https://numpy.org/neps/nep-0029-deprecation_policy.html).

This project supports:

* All minor versions of Python released 42 months prior to the project, and at minimum the two latest minor versions.
* All minor versions of numpy released in the 24 months prior to the project, and at minimum the last three minor versions.
* Qt5 5.15, and Qt6 6.2+

Currently this means:

* Python 3.8+
* Qt 5.15, 6.2+
* [PyQt5](https://www.riverbankcomputing.com/software/pyqt/),
  [PyQt6](https://www.riverbankcomputing.com/software/pyqt/),
  [PySide2](https://wiki.qt.io/Qt_for_Python), or
  [PySide6](https://wiki.qt.io/Qt_for_Python)
* [`numpy`](https://github.com/numpy/numpy) 1.20+

### Optional added functionalities

Through 3rd part libraries, additional functionality may be added to PyQtGraph, see the table below for a summary.

| Library        | Added functionality |
|----------------|-|
| [`scipy`]      | <ul><li> Image processing through [`ndimage`]</li><li> Data array filtering through [`signal`] </li><ul> |
| [`pyopengl`]   | <ul><li> 3D graphics </li><li> Faster image processing </li><li>Note: on macOS Big Sur only works with python 3.9.1+</li></ul> |
| [`h5py`]       | <ul><li> Export in hdf5 format </li></ul> |
| [`colorcet`]   | <ul><li> Add a collection of perceptually uniform colormaps </li></ul> |
| [`matplotlib`] | <ul><li> Export of PlotItem in matplotlib figure </li><li> Add matplotlib collection of colormaps </li></ul> |
| [`cupy`]       | <ul><li> CUDA-enhanced image processing </li><li> Note: On Windows, CUDA toolkit must be >= 11.1 </li></ul> |
| [`numba`]      | <ul><li> Faster image processing </li></ul> |
| [`jupyter_rfb`]| <ul><li> Jupyter Notebook support </li> <li> [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pyqtgraph/pyqtgraph/HEAD?urlpath=%2Flab%2Ftree%2Fpyqtgraph%2Fexamples%2Fnotebooks) </li> </ul> |

[`scipy`]: https://github.com/scipy/scipy
[`ndimage`]: https://docs.scipy.org/doc/scipy/reference/ndimage.html
[`signal`]: https://docs.scipy.org/doc/scipy/reference/signal.html
[`pyopengl`]: https://github.com/mcfletch/pyopengl
[`h5py`]: https://github.com/h5py/h5py
[`colorcet`]: https://github.com/holoviz/colorcet
[`matplotlib`]: https://github.com/matplotlib/matplotlib
[`numba`]: https://github.com/numba/numba
[`cupy`]: https://docs.cupy.dev/en/stable/install.html
[`jupyter_rfb`]: https://github.com/vispy/jupyter_rfb

Support
-------

* Report issues on the [GitHub issue tracker](https://github.com/pyqtgraph/pyqtgraph/issues)
* Post questions to
  * [mailing list / forum](https://groups.google.com/forum/?fromgroups#!forum/pyqtgraph)
  * [StackOverflow](https://stackoverflow.com/questions/tagged/pyqtgraph)
  * [GitHub Discussions](https://github.com/pyqtgraph/pyqtgraph/discussions)
  * [Python Discord](https://discord.com/channels/267624335836053506/898139460821192724)

Installation Methods
--------------------

* From PyPI
  * Last released version: `pip install pyqtgraph`
  * Latest development version: `pip install git+https://github.com/pyqtgraph/pyqtgraph@master`
* From conda
  * Last released version: `conda install -c conda-forge pyqtgraph`
* To install system-wide from source distribution: `python setup.py install`
* Many linux package repositories have release versions.
* To use with a specific project, simply copy the PyQtGraph subdirectory
  anywhere that is importable from your project.

Documentation
-------------

The official documentation lives at [pyqtgraph.readthedocs.io](https://pyqtgraph.readthedocs.io)

The easiest way to learn PyQtGraph is to browse through the examples; run `python -m pyqtgraph.examples` to launch the examples application.

Used By
-------

Here is a partial listing of some of the applications that make use of PyQtGraph!

* [ACQ4](https://github.com/acq4/acq4)
* [Antenna Array Analysis](https://github.com/rookiepeng/antenna-array-analysis)
* [argos](https://github.com/titusjan/argos)
* [Atomize](https://github.com/Anatoly1010/Atomize)
* [EnMAP-Box](https://enmap-box.readthedocs.io)
* [EO Time Series Viewer](https://eo-time-series-viewer.readthedocs.io)
* [ephyviewer](https://ephyviewer.readthedocs.io)
* [Exo-Striker](https://github.com/3fon3fonov/exostriker)
* [GraPhysio](https://github.com/jaj42/GraPhysio)
* [HussariX](https://github.com/sem-geologist/HussariX)
* [Joulescope](https://www.joulescope.com/)
* [MaD GUI](https://github.com/mad-lab-fau/mad-gui)
* [neurotic](https://neurotic.readthedocs.io)
* [Orange3](https://orangedatamining.com/)
* [PatchView](https://github.com/ZeitgeberH/patchview)
* [pyplotter](https://github.com/pyplotter/pyplotter)
* [PyMeasure](https://github.com/pymeasure/pymeasure)
* [PySpectra](http://hasyweb.desy.de/services/computing/Spock/node138.html)
* [rapidtide](https://rapidtide.readthedocs.io/en/latest/)
* [Semi-Supervised Semantic Annotator](https://gitlab.com/s3a/s3a)
* [STDF-Viewer](https://github.com/noonchen/STDF-Viewer)


Do you use PyQtGraph in your own project, and want to add it to the list?  Submit a pull request to update this listing!
