from distutils.core import setup
import distutils.dir_util
import os

## generate list of all sub-packages
path = os.path.abspath(os.path.dirname(__file__))
n = len(path.split(os.path.sep))
subdirs = [i[0].split(os.path.sep)[n:] for i in os.walk(os.path.join(path, 'pyqtgraph')) if '__init__.py' in i[2]]
all_packages = ['.'.join(p) for p in subdirs] + ['pyqtgraph.examples']


## Make sure build directory is clean before installing
buildPath = os.path.join(path, 'build')
if os.path.isdir(buildPath):
    distutils.dir_util.remove_tree(buildPath)

setup(name='pyqtgraph',
    version='',
    description='Scientific Graphics and GUI Library for Python',
    long_description="""\
PyQtGraph is a pure-python graphics and GUI library built on PyQt4/PySide and
numpy. 

It is intended for use in mathematics / scientific / engineering applications.
Despite being written entirely in python, the library is very fast due to its
heavy leverage of numpy for number crunching, Qt's GraphicsView framework for
2D display, and OpenGL for 3D display.
""",
    license='MIT',
    url='http://www.pyqtgraph.org',
    author='Luke Campagnola',
    author_email='luke.campagnola@gmail.com',
    packages=all_packages,
    package_dir={'pyqtgraph.examples': 'examples'},  ## install examples along with the rest of the source
    #package_data={'pyqtgraph': ['graphicsItems/PlotItem/*.png']},
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Environment :: Other Environment",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: User Interfaces",
        ],
    install_requires = [
        'numpy',
        'scipy',
        ],
)

