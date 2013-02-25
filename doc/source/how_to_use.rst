How to use pyqtgraph
====================

There are a few suggested ways to use pyqtgraph:
    
* From the interactive shell (python -i, ipython, etc)
* Displaying pop-up windows from an application
* Embedding widgets in a PyQt application



Command-line use
----------------

Pyqtgraph makes it very easy to visualize data from the command line. Observe::
    
    import pyqtgraph as pg
    pg.plot(data)   # data can be a list of values or a numpy array

The example above would open a window displaying a line plot of the data given. The call to :func:`pg.plot <pyqtgraph.plot>` returns a handle to the :class:`plot widget <pyqtgraph.PlotWidget>` that is created, allowing more data to be added to the same window. **Note:** interactive plotting from the python prompt is only available with PyQt; PySide does not run the Qt event loop while the interactive prompt is running. If you wish to use pyqtgraph interactively with PySide, see the 'console' :ref:`example <examples>`.

Further examples::
    
    pw = pg.plot(xVals, yVals, pen='r')  # plot x vs y in red
    pw.plot(xVals, yVals2, pen='b')
    
    win = pg.GraphicsWindow()  # Automatically generates grids with multiple items
    win.addPlot(data1, row=0, col=0)
    win.addPlot(data2, row=0, col=1)
    win.addPlot(data3, row=1, col=0, colspan=2)

    pg.show(imageData)  # imageData must be a numpy array with 2 to 4 dimensions
    
We're only scratching the surface here--these functions accept many different data formats and options for customizing the appearance of your data.


Displaying windows from within an application
---------------------------------------------

While I consider this approach somewhat lazy, it is often the case that 'lazy' is indistinguishable from 'highly efficient'. The approach here is simply to use the very same functions that would be used on the command line, but from within an existing application. I often use this when I simply want to get a immediate feedback about the state of data in my application without taking the time to build a user interface for it.


Embedding widgets inside PyQt applications
------------------------------------------

For the serious application developer, all of the functionality in pyqtgraph is available via :ref:`widgets <api_widgets>` that can be embedded just like any other Qt widgets. Most importantly, see: :class:`PlotWidget <pyqtgraph.PlotWidget>`, :class:`ImageView <pyqtgraph.ImageView>`, :class:`GraphicsLayoutWidget <pyqtgraph.GraphicsLayoutWidget>`, and :class:`GraphicsView <pyqtgraph.GraphicsView>`. Pyqtgraph's widgets can be included in Designer's ui files via the "Promote To..." functionality:
    
#. In Designer, create a QGraphicsView widget ("Graphics View" under the "Display Widgets" category).
#. Right-click on the QGraphicsView and select "Promote To...".
#. Under "Promoted class name", enter the class name you wish to use ("PlotWidget", "GraphicsLayoutWidget", etc).
#. Under "Header file", enter "pyqtgraph".
#. Click "Add", then click "Promote".

See the designer documentation for more information on promoting widgets.


PyQt and PySide
---------------

Pyqtgraph supports two popular python wrappers for the Qt library: PyQt and PySide. Both packages provide nearly identical 
APIs and functionality, but for various reasons (discussed elsewhere) you may prefer to use one package or the other. When
pyqtgraph is first imported, it automatically determines which library to use by making the fillowing checks:
    
#. If PyQt4 is already imported, use that
#. Else, if PySide is already imported, use that
#. Else, attempt to import PyQt4
#. If that import fails, attempt to import PySide. 

If you have both libraries installed on your system and you wish to force pyqtgraph to use one or the other, simply
make sure it is imported before pyqtgraph::
    
    import PySide  ## this will force pyqtgraph to use PySide instead of PyQt4
    import pyqtgraph as pg
