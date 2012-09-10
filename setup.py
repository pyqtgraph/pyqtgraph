from distutils.core import setup

setup(name='pyqtgraph',
    version='',
    description='Scientific Graphics and GUI Library for Python',
    long_description="PyQtGraph is a pure-python graphics and GUI library built on PyQt4 and numpy. It is intended for use in mathematics / scientific / engineering applications. Despite being written entirely in python, the library is very fast due to its heavy leverage of numpy for number crunching and Qt's GraphicsView framework for fast display.",
    license='MIT',
    author='Luke Campagnola',
    author_email='luke.campagnola@gmail.com',
    url='',
    packages=['pyqtgraph', 'pyqtgraph.console', 'pyqtgraph.graphicsItems', 'pyqtgraph.widgets', 'pyqtgraph.metaarray', 'pyqtgraph.parametertree', 'pyqtgraph.flowchart', 'pyqtgraph.imageview', 'pyqtgraph.dockarea', 'pyqtgraph.examples', 'pyqtgraph.canvas', 'pyqtgraph.exporters', 'pyqtgraph.GraphicsScene', 'pyqtgraph.multiprocess', 'pyqtgraph.opengl'],
    package_dir = {'pyqtgraph': '.'},
    package_data={'pyqtgraph': ['graphicsItems/PlotItem/*.png']},
)




#Package: python-pyqtgraph
#Version: 196
#Section: custom
#Priority: optional
#Architecture: all
#Essential: no
#Installed-Size: 4652
#Maintainer: Luke Campagnola <luke.campagnola@gmail.com>
#Homepage: http://luke.campagnola.me/code/pyqtgraph
#Depends: python (>= 2.7), python-qt4 | python-pyside, python-scipy
#Suggests: python-opengl, python-qt4-gl
#Description: Scientific Graphics and GUI Library for Python
 #PyQtGraph is a pure-python graphics and GUI library built on PyQt4 and numpy. It is intended for use in mathematics / scientific / engineering applications. Despite being written entirely in python, the library is very fast due to its heavy leverage of numpy for number crunching and Qt's GraphicsView framework for fast display.
