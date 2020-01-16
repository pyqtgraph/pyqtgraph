Installation
============

There are many different ways to install pyqtgraph, depending on your needs:

* The most common way to install pyqtgraph is with pip::
    
      $ pip install pyqtgraph
    
  Some users may need to call ``pip3`` instead. This method should work on
  all platforms. 
* To get access to the very latest features and bugfixes you have three choice::
  
  1. Clone pyqtgraph from github::
      
      $ git clone https://github.com/pyqtgraph/pyqtgraph
    
    Now you can install pyqtgraph from the source::

      $ python setup.py install
      
  2. Directly install from GitHub repo::

      $ pip install git+git://github.com/pyqtgraph/pyqtgraph.git@develop

    You can change to ``develop`` of the above command to the branch 
    name or the commit you prefer.
  
  3.
    You can simply place the pyqtgraph folder someplace importable, such as
    inside the root of another project. PyQtGraph does not need to be "built" or
    compiled in any way.

* Packages for pyqtgraph are also available in a few other forms:
    
  * **Anaconda**: ``conda install pyqtgraph``
  * **Debian, Ubuntu, and similar Linux:** Use ``apt install python-pyqtgraph`` or
    download the .deb file linked at the top of the pyqtgraph web page.
  * **Arch Linux:** has packages (thanks windel). (https://aur.archlinux.org/packages.php?ID=62577)
  * **Windows:** Download and run the .exe installer file linked at the top of the pyqtgraph web page.


Requirements
============

PyQtGraph depends on:
    
* Python 2.7 or Python 3.x
* A Qt library such as PyQt4, PyQt5, PySide, or PySide2
* numpy

The easiest way to meet these dependencies is with ``pip`` or with a scientific python
distribution like Anaconda.

.. _pyqtgraph: http://www.pyqtgraph.org/
