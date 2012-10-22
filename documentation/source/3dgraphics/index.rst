Pyqtgraph's 3D Graphics System
==============================

The 3D graphics system in pyqtgraph is composed of a :class:`view widget <pyqtgraph.opengl.GLViewWidget>` and 
several graphics items (all subclasses of :class:`GLGraphicsItem <pyqtgraph.opengl.GLGraphicsItem>`) which 
can be added to a view widget.

**Note:** use of this system requires python-opengl bindings. Linux users should install the python-opengl
packages from their distribution. Windows/OSX users can download from `<http://pyopengl.sourceforge.net>`_.

Contents:

.. toctree::
    :maxdepth: 2

    glviewwidget

    glgriditem
    glmeshitem
    glvolumeitem
    glimageitem
    glaxisitem
    glgraphicsitem
    meshdata

