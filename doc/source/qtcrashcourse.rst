Qt Crash Course
===============

PyQtGraph makes extensive use of Qt for generating nearly all of its visual output and interfaces. Qt's documentation is very well written and we encourage all pyqtgraph developers to familiarize themselves with it. The purpose of this section is to provide an introduction to programming with Qt (using either PyQt or PySide) for the pyqtgraph developer.

QWidgets and Layouts
--------------------

A Qt GUI is almost always composed of a few basic components:
    
* A window. This is often provided by QMainWindow, but note that all QWidgets can be displayed in their window by simply calling widget.show() if the widget does not have a parent. 
* Multiple QWidget instances such as QPushButton, QLabel, QComboBox, etc. 
* QLayout instances (optional, but strongly encouraged) which automatically manage the positioning of widgets to allow the GUI to resize in a usable way.

PyQtGraph fits into this scheme by providing its own QWidget subclasses to be inserted into your GUI.


Example::
    
    from PyQt5 import QtGui  # (the example applies equally well to PySide2)
    import pyqtgraph as pg
        
    ## Always start by initializing Qt (only once per application)
    app = QtGui.QApplication([])

    ## Define a top-level widget to hold everything
    w = QtGui.QWidget()

    ## Create some widgets to be placed inside
    btn = QtGui.QPushButton('press me')
    text = QtGui.QLineEdit('enter text')
    listw = QtGui.QListWidget()
    plot = pg.PlotWidget()

    ## Create a grid layout to manage the widgets size and position
    layout = QtGui.QGridLayout()
    w.setLayout(layout)

    ## Add widgets to the layout in their proper positions
    layout.addWidget(btn, 0, 0)   # button goes in upper-left
    layout.addWidget(text, 1, 0)   # text edit goes in middle-left
    layout.addWidget(listw, 2, 0)  # list widget goes in bottom-left
    layout.addWidget(plot, 0, 1, 3, 1)  # plot goes on right side, spanning 3 rows

    ## Display the widget as a new window
    w.show()

    ## Start the Qt event loop
    app.exec_()

More complex interfaces may be designed graphically using Qt Designer, which allows you to simply drag widgets into your window to define its appearance.


Naming Conventions
------------------

Virtually every class in pyqtgraph is an extension of base classes provided by Qt. When reading the documentation, remember that all of Qt's classes start with the letter 'Q', whereas pyqtgraph's classes do not. When reading through the methods for any class, it is often helpful to see which Qt base classes are used and look through the Qt documentation as well.

Most of Qt's classes define signals which can be difficult to tell apart from regular methods. Almost all signals explicity defined by pyqtgraph are named beginning with 'sig' to indicate that these signals are not defined at the Qt level.

In most cases, classes which end in 'Widget' are subclassed from QWidget and can therefore be used as a GUI element in a Qt window. Classes which end in 'Item' are subclasses of QGraphicsItem and can only be displayed within a QGraphicsView instance (such as GraphicsLayoutWidget or PlotWidget). 


Signals, Slots, and Events
--------------------------

[ to be continued.. please post a request on the pyqtgraph forum if you'd like to read more ]

Qt detects and reacts to user interaction by executing its *event loop*. 

 - what happens in the event loop?
 - when do I need to use QApplication.exec_() ?
 - what control do I have over event loop execution? (QApplication.processEvents)


GraphicsView and GraphicsItems
------------------------------

More information about the architecture of Qt GraphicsView:
http://qt-project.org/doc/qt-4.8/graphicsview.html


Coordinate Systems and Transformations
--------------------------------------

More information about the coordinate systems in Qt GraphicsView:
http://qt-project.org/doc/qt-4.8/graphicsview.html#the-graphics-view-coordinate-system


Mouse and Keyboard Input
------------------------




QTimer, Multi-Threading
-----------------------


Multi-threading vs Multi-processing in Qt
-----------------------------------------
