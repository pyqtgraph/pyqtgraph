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

Events Overview
^^^^^^^^^^^^^^^
Understanding events in a Qt application is fundamental before delving into specific input handling such as mouse and keyboard:

- Events in Qt are conceptualized as user interactions with the application, each represented by an event object (QEvent).
- Various types of events correspond to different user interactions.
- Event objects encapsulate details concerning the specific occurrence.
- Dispatched to designated event handlers within the widget where the interaction occurs, these events allow for customizable responses.
- Handlers may be extended or redefined to modify widget response to interactions.

Mouse Events
^^^^^^^^^^^^
Mouse Events
^^^^^^^^^^^^
Interactions such as clicks, movements, and button releases are managed by overriding methods including :meth:`QtWidget.mousePressEvent <https://doc.qt.io/qt-6/qwidget.html#mousePressEvent>`, :meth:`QtWidget.mouseReleaseEvent <https://doc.qt.io/qt-6/qwidget.html#mouseReleaseEvent>`, :meth:`QtWidget.mouseDoubleClickEvent <https://doc.qt.io/qt-6/qwidget.html#mouseDoubleClickEvent>`, and :meth:`QtWidget.mouseMoveEvent <https://doc.qt.io/qt-6/qwidget.html#mouseMoveEvent>`.

Integration with PyQtGraph
~~~~~~~~~~~~~~~~~~~~~~~~~~
PyQtGraph utilizes QWidget subclasses to present graphics and plots. Consequently, the event-handling methods discussed can be directly integrated into PyQtGraph widgets. This integration enables sophisticated interactive features in applications that leverage PyQtGraph for visual data representation.

Example: Handling Mouse Clicks in a PlotWidget::

    from PyQt6.QtWidgets import QApplication, QMainWindow
    from PyQt6.QtCore import Qt

    # Should work with PyQt5 / PySide2 / PySide6 as well

    class MainWindow(QMainWindow):
    def __init__(self):
    super().__init__()
    # Sets the Title of the window
    self.setWindowTitle('Mouse and Keyboard Event Demo')
    # Sets the position and size of the window
    self.setGeometry(100, 100, 400, 300)

    def mousePressEvent(self, event):
    # This method checks if the left mouse button was pressed on the widget
    # and prints the position of the click.
    if event.button() == Qt.MouseButton.LeftButton:
    print("Left mouse button pressed at:", event.position())

    # Initialize the QApplication
    app = QApplication([])
    window = MainWindow()
    window.show()
    # Start the event loop
    app.exec()


This code snippet demonstrates initializing a basic PyQt6 application that responds to a left mouse button click, illustrating the practical application of handling mouse events in a PyQtGraph environment.

Keyboard Events
^^^^^^^^^^^^^^^
Keyboard inputs are similarly handled by overriding :meth:`QtWidget.keyPressEvent <https://doc.qt.io/qt-6/qwidget.html#keyPressEvent>` and :meth:`QtWidget.keyReleaseEvent <https://doc.qt.io/qt-6/qwidget.html#keyReleaseEvent>`, allowing applications to react to various keystrokes and facilitate shortcuts and other interactions.


Integration with PyQtGraph
~~~~~~~~~~~~~~~~~~~~~~~~~~

Example: Handling Keyboard Inputs:: 

    from PyQt6.QtWidgets import QApplication, QMainWindow
    from PyQt6.QtCore import Qt

    # Should work with PyQt5 / PySide2 / PySide6 as well

    class MainWindow(QMainWindow):
    def __init__(self):
    super().__init__()
    # Sets the title of the window
    self.setWindowTitle('Keyboard Input Tracker')
    # Sets the position and size of the window
    self.setGeometry(100, 100, 400, 300)

    def keyPressEvent(self, event):
    # Checks if a specific key was pressed
    if event.key() == Qt.Key.Key_Escape:
    print("Escape key was pressed.")
    elif event.key() == Qt.Key.Key_Space:
    print("Space bar was pressed.")
    else:
    # The 'event.text()' method retrieves the character or characters 
    # associated with the key press, and then prints it to the console.
    print(f"Key pressed: {event.text()}")

    # Initialize the QApplication
    app = QApplication([])
    window = MainWindow()
    window.show()
    # Start the event loop
    app.exec()


Event Propagation
^^^^^^^^^^^^^^^^^
In PyQt, when an event is not handled by a widget, or the widget explicitly decides against handling it, the event is propagated to its parent widget. This process, commonly referred to as "bubbling," continues upward through the nested widgets until the event is either handled or reaches the main window.

It is facilitated by methods such as :meth:`QEvent.accept <https://doc.qt.io/qt-6/qevent.html#accept>` and :meth:`QEvent.ignore <https://doc.qt.io/qt-6/qevent.html#ignore>`, which allow developers to exert precise control over the event flow.

Example: Custom Event Handling ::

    class CustomButton(QPushButton):
    def mousePressEvent(self, event):
    event.accept() # The event is marked as handled, preventing further propagation
    # Alternatively: 
    event.ignore() # the event can be marked as unhandled, allowing it to propagate further


QTimer, Multi-Threading
-----------------------


Multi-threading vs Multi-processing in Qt
-----------------------------------------
