.. currentmodule:: pyqtgraph

.. _apiref_config:

Global Configuration Options
============================

PyQtGraph has several global configuration options that allow you to change its
default behavior. These can be accessed using the :func:`setConfigOptions` and 
:func:`getConfigOption` functions:
    
================== =================== ================== ================================================================================
**Option**         **Type**            **Default**
leftButtonPan      bool                True               If True, dragging the left mouse button over a ViewBox
                                                          causes the view to be panned. If False, then dragging
                                                          the left mouse button draws a rectangle that the 
                                                          ViewBox will zoom to.
foreground         See :func:`mkColor` 'd'                Default foreground color for text, lines, axes, etc.
background         See :func:`mkColor` 'k'                Default background for :class:`GraphicsView`.
antialias          bool                False              Enabling antialiasing causes lines to be drawn with 
                                                          smooth edges at the cost of reduced performance.
imageAxisOrder     str                 'col-major'        For 'row-major', image data is expected in the standard row-major 
                                                          (row, col) order. For 'col-major', image data is expected in
                                                          reversed column-major (col, row) order.
                                                          The default is 'col-major' for backward compatibility, but this may
                                                          change in the future.
editorCommand      str or None         None               Command used to invoke code editor from ConsoleWidget.
exitCleanup        bool                True               Attempt to work around some exit crash bugs in PyQt and PySide.
useOpenGL          bool                False              Enable OpenGL in GraphicsView.
useCupy            bool                False              Use cupy to perform calculations on the GPU. Only currently applies to
                                                          ImageItem and its associated functions.
enableExperimental bool                False              Enable experimental features (the curious can search for this key in the code).
crashWarning       bool                False              If True, print warnings about situations that may result in a crash.
================== =================== ================== ================================================================================


.. autofunction:: pyqtgraph.setConfigOptions

.. autofunction:: pyqtgraph.getConfigOption

