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

The example above would open a window displaying a line plot of the data given. I don't think it could reasonably be any simpler than that. The call to pg.plot returns a handle to the plot widget that is created, allowing more data to be added to the same window.

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

For the serious application developer, all of the functionality in pyqtgraph is available via widgets that can be embedded just like any other Qt widgets. Most importantly, see: PlotWidget, ImageView, GraphicsView, GraphicsLayoutWidget. Pyqtgraph's widgets can be included in Designer's ui files via the "Promote To..." functionality.

