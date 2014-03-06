3D Graphics API
==============================

The 3D graphics system in PyQtGraph is composed of a :class:`~pyqtgraph.opengl.GLViewWidget` and 
several graphics items (all subclasses of :class:`~pyqtgraph.opengl.GLGraphicsItem.GLGraphicsItem`) which 
can be added to a view widget.

.. warning::
    **pyqtgraph.opengl** is based on the deprecated OpenGL fixed-function pipeline. Although it is
    currently a functioning system, it is likely to be superceded in the future by `VisPy <http://vispy.org>`_.

.. note:: Use of this system requires python-opengl bindings. Linux users should install the python-opengl
          packages from their distribution. Windows/OSX users can download from `<http://pyopengl.sourceforge.net>`_.

.. seealso::
       :ref:`3D_graphics_guide` guide
    
.. rubric:: Contents:

.. toctree::
    :maxdepth: 2

    glviewwidget

    glgriditem
    glsurfaceplotitem
    glvolumeitem
    glimageitem
    glmeshitem
    gllineplotitem
    glaxisitem
    glgraphicsitem
    glscatterplotitem
    meshdata

