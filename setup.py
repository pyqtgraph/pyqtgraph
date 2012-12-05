from distutils.core import setup
import os

## generate list of all sub-packages
subdirs = [['pyqtgraph'] + i[0].split(os.path.sep)[1:] for i in os.walk('.') if '__init__.py' in i[2]]
subdirs = filter(lambda p: len(p) == 1 or p[1] != 'build', subdirs)
all_packages = ['.'.join(p) for p in subdirs]

setup(name='pyqtgraph',
    version='',
    description='Scientific Graphics and GUI Library for Python',
    long_description="PyQtGraph is a pure-python graphics and GUI library built on PyQt4 and numpy. It is intended for use in mathematics / scientific / engineering applications. Despite being written entirely in python, the library is very fast due to its heavy leverage of numpy for number crunching and Qt's GraphicsView framework for fast display.",
    license='MIT',
    url='http://www.pyqtgraph.org',
    author='Luke Campagnola',
    author_email='luke.campagnola@gmail.com',
    packages=all_packages,
    package_dir = {'pyqtgraph': '.'},
    package_data={'pyqtgraph': ['graphicsItems/PlotItem/*.png']},
)

