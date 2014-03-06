About PyQtGraph
===================

PyQtGraph is a graphics and user interface library for :term:`Python` that provides functionality commonly required in engineering and science applications.
Primary goals are 

#. Provide fast, interactive graphics for displaying data such as plots, video, etc 
#. Provide tools to aid in rapid application development, for example, property trees such as used in Qt Designer

PyQtGraph makes heavy use of the :term:`Qt` platform for its high-performance graphics, via :term:`PyQt` or :term:`PySide` :term:`Python` bindings, and using :term:`numpy`/:term:`scipy` for heavy number crunching. In particular, PyQtGraph uses Qt's GraphicsView framework which is a highly capable graphics system on its own. PyQtGraph bring optimized and simplified primitives to this framework to allow data visualization with minimal effort. 

It is known to run on Linux, Windows, and OSX (:ref:`install`)


Features
-------------------
* Basic 2D plotting in interactive view boxes
    * Line and scatter plots
    * Data can be panned/scaled by mouse
    * Fast drawing for realtime data display and interaction
    
    .. image:: http://pyqtgraph.org/images/plotting.png
       :height: 300
    
* Image display with interactive lookup tables and level control
    * Displays most data types (int or float; any bit depth; RGB, RGBA, or luminance)
    * Functions for slicing multidimensional images at arbitrary angles (great for MRI data)
    * Rapid update for video display or realtime interaction
    
    .. image:: http://pyqtgraph.org/images/data_slicing.png
       :height: 300
       
* 3D graphics system (requires Python-OpenGL bindings)
    * Volumetric data rendering
    * 3D surface and scatter plots
    * Mesh rendering with isosurface generation
    * Interactive viewports rotate/zoom with mouse
    * Basic 3D scenegraph for easier programming
    
    .. image:: http://pyqtgraph.org/images/pyqtgraph-3d.png
       :height: 300
    
* Data selection/marking and region-of-interest controls
    * Interactively mark vertical/horizontal locations and regions in plots
    * Widgets for selecting arbitrary regions from images and automatically slicing data to match
    
    .. image:: http://pyqtgraph.org/images/screenshot3.png
       :height: 300
    
* Easy to generate new graphics
    * 2D graphics use Qt's GraphicsView framework which is highly capable and mature.
    * 3D graphics use OpenGL
    * All graphics use a scenegraph for managing items; new graphics items are simple to create.
    
* Library of widgets and modules useful for science/engineering applications
    * Flowchart widget for interactive prototyping.
    * Interface similar to LabView (nodes connected by wires).
    * Parameter tree widget for displaying/editing hierarchies of parameters
    * (similar to those used by most GUI design applications).
    * Interactive python console with exception catching.
    * Great for debugging/introspection as well as advanced user interaction.
    * Multi-process control allowing remote plotting, Qt signal connection across processes, and very simple in-line parallelization.
    * Dock system allowing the user to rearrange GUI components. 
    * Similar to Qt's dock system but a little more flexible and programmable.
    * Color gradient editor
    * SpinBox with SI-unit display and logarithmic stepping
    
    .. image:: http://pyqtgraph.org/images/flowchart.png
       :height: 300
    
    
* See :ref:`examples` for a demo


How does it compare to...
-------------------------

* matplotlib: For plotting, pyqtgraph is not nearly as complete/mature as matplotlib, but runs much faster. Matplotlib is more aimed toward making publication-quality graphics, whereas pyqtgraph is intended for use in data acquisition and analysis applications. Matplotlib is more intuitive for matlab programmers; pyqtgraph is more intuitive for python/qt programmers. Matplotlib (to my knowledge) does not include many of pyqtgraph's features such as image interaction, volumetric rendering, parameter trees, flowcharts, etc.

* pyqwt5: About as fast as pyqwt5, but not quite as complete for plotting functionality. Image handling in pyqtgraph is much more complete (again, no ROI widgets in qwt). Also, pyqtgraph is written in pure python, so it is more portable than pyqwt, which often lags behind pyqt in development (I originally used pyqwt, but decided it was too much trouble to rely on it as a dependency in my projects). Like matplotlib, pyqwt (to my knowledge) does not include many of pyqtgraph's features such as image interaction, volumetric rendering, parameter trees, flowcharts, etc.

(My experience with these libraries is somewhat outdated; please correct me if I am wrong here)
