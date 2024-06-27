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

    from PyQt6 import QtWidgets  # Should work with PyQt5 / PySide2 / PySide6 as well
    import pyqtgraph as pg
    
    ## Always start by initializing Qt (only once per application)
    app = QtWidgets.QApplication([])

    ## Define a top-level widget to hold everything
    w = QtWidgets.QWidget()
    w.setWindowTitle('PyQtGraph example')

    ## Create some widgets to be placed inside
    btn = QtWidgets.QPushButton('press me')
    text = QtWidgets.QLineEdit('enter text')
    listw = QtWidgets.QListWidget()
    plot = pg.PlotWidget()

    ## Create a grid layout to manage the widgets size and position
    layout = QtWidgets.QGridLayout()
    w.setLayout(layout)

    ## Add widgets to the layout in their proper positions
    layout.addWidget(btn, 0, 0)  # button goes in upper-left
    layout.addWidget(text, 1, 0)  # text edit goes in middle-left
    layout.addWidget(listw, 2, 0)  # list widget goes in bottom-left
    layout.addWidget(plot, 0, 1, 3, 1)  # plot goes on right side, spanning 3 rows
    ## Display the widget as a new window
    w.show()

    ## Start the Qt event loop
    app.exec()  # or app.exec_() for PyQt5 / PySide2


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


PyQt handles mouse and keyboard inputs as events, which are processed by the application's event loop. Each widget in PyQt can respond to these inputs by implementing specific event handlers.

Mouse Events
^^^^^^^^^^^^
These include clicks, movements, and mouse button releases. In PyQt, you can handle these events by overriding methods such as ``mousePressEvent``, ``mouseReleaseEvent``, and ``mouseMoveEvent``.

Keyboard Events
^^^^^^^^^^^^^^^
Similarly, keyboard events can be managed by overriding methods like ``keyPressEvent`` and ``keyReleaseEvent``. These methods allow you to react to different keys being pressed, providing a way to implement shortcuts and other keyboard interactions within your application.

Integration with PyQtGraph
^^^^^^^^^^^^^^^^^^^^^^^^^^
PyQtGraph utilizes QWidget subclasses to present graphics and plots. Consequently, the event-handling methods discussed can be directly integrated into PyQtGraph widgets. This integration enables sophisticated interactive features in applications that leverage PyQtGraph for visual data representation.

Example: Handling Mouse Clicks in a PlotWidget::

    from PyQt6.QtWidgets import QApplication, QMainWindow
    from PyQt6.QtCore import Qt

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('Mouse and Keyboard Event Demo')
            self.setGeometry(100, 100, 400, 300)

        def mousePressEvent(self, event):
            # This method checks if the mouse was pressed on the widget
            if event.button() == Qt.MouseButton.LeftButton:
                print("Left mouse button pressed at:", event.position())

    # Initialize the QApplication
    app = QApplication([])
    window = MainWindow()
    window.show()

    # Start the event loop
    app.exec()

This code snippet demonstrates initializing a basic PyQt6 application that responds to a left mouse button click, illustrating the practical application of handling mouse events in a PyQtGraph environment.

Example:: 
    from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import Qt
import pyqtgraph as pg

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PyQtGraph Pan Example')  # Sets the window title
        self.setGeometry(100, 100, 800, 600)  # Sets the window size

        # Initialize a PlotWidget from pyqtgraph and set it as the central widget
        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)

        # Add some data to plot
        self.plot_widget.plot([1, 2, 3, 4, 5], [5, 6, 10, 8, 7])

    def mousePressEvent(self, event):
        # Check if the left mouse button was pressed
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_position = event.position()  # Store the last mouse position

    def mouseMoveEvent(self, event):
        # Ensure the last mouse position is defined
        if not hasattr(self, 'last_mouse_position'):
            return
        
        current_position = event.position()
        delta = current_position - self.last_mouse_position

        # Translate the plot view by the amount of mouse movement
        self.plot_widget.plotItem.getViewBox().translateBy(x=-delta.x(), y=-delta.y())
        self.last_mouse_position = current_position  # Update the last mouse position for the next move event

# Initialize the QApplication
app = QApplication([])
window = MainWindow()
window.show()

# Start the event loop
app.exec()



QTimer, Multi-Threading
-----------------------


Multi-threading vs Multi-processing in Qt
-----------------------------------------
