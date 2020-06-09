Installation
============

PyQtGraph depends on:

* Python 2.7 or Python 3.x
* A Qt library such as PyQt4, PyQt5, PySide, or PySide2
* numpy

The easiest way to meet these dependencies is with ``pip`` or with a scientific
python distribution like Anaconda.

There are many different ways to install pyqtgraph, depending on your needs:

pip
---

The most common way to install pyqtgraph is with pip::

    $ pip install pyqtgraph

Some users may need to call ``pip3`` instead. This method should work on all
platforms.

conda
-----

pyqtgraph is on the default Anaconda channel::

    $ conda install pyqtgraph

It is also available in the conda-forge channel::

    $ conda install -c conda-forge pyqtgraph

From Source
-----------

To get access to the very latest features and bugfixes you have three choices:

1. Clone pyqtgraph from github::

    $ git clone https://github.com/pyqtgraph/pyqtgraph
    $ cd pyqtgraph

   Now you can install pyqtgraph from the source::

    $ pip install .

2. Directly install from GitHub repo::

    $ pip install git+git://github.com/pyqtgraph/pyqtgraph.git@develop

   You can change ``develop`` of the above command to the branch name or the
   commit you prefer.

3. You can simply place the pyqtgraph folder someplace importable, such as
   inside the root of another project. PyQtGraph does not need to be "built" or
   compiled in any way.

Other Packages
--------------

Packages for pyqtgraph are also available in a few other forms:

* **Debian, Ubuntu, and similar Linux:** Use ``apt install python-pyqtgraph`` or
  download the .deb file linked at the top of the pyqtgraph web page.
* **Arch Linux:** https://www.archlinux.org/packages/community/any/python-pyqtgraph/
* **Windows:** Download and run the .exe installer file linked at the top of the
  pyqtgraph web page: http://pyqtgraph.org
