Qt Crash Course
===============

PyQtGraph makes extensive use of Qt for generating nearly all of its visual output and
interfaces. Qt's documentation is very well written and we encourage all pyqtgraph
developers to familiarize themselves with it. The purpose of this section is to provide
an introduction to programming with Qt (using either PyQt or PySide) for the pyqtgraph
developer.

QWidgets and Layouts
--------------------

A Qt GUI is almost always composed of a few basic components:
    
* A window. This is often provided by QMainWindow, but note that all QWidgets can be
  displayed in their window by simply calling :meth:`QWidget.show <QWidget.show>` if the
  widget does not have a parent. 
* Multiple :class:`QWidget` instances such as :class:`QPushButton`, :class:`QLabel`,
  :class:`QComboBox`, etc. 
* :class:`QLayout` instances (optional, but strongly encouraged) which automatically
  manage the positioning of widgets to allow the GUI to resize in a usable way.

PyQtGraph fits into this scheme by providing its own :class:`QWidget` subclasses to be
inserted into your GUI.

.. code-block:: python
  :caption: Example PyQtGraph Integration Into Qt

  from PyQt6 import QtWidgets  # Should work with PyQt5 / PySide2 / PySide6 as well
  import pyqtgraph as pg
  
  # Always start by initializing Qt (only once per application)
  app = QtWidgets.QApplication([])
  
  # Define a top-level widget to hold everything
  w = QtWidgets.QWidget()
  w.setWindowTitle('PyQtGraph example')
  
  # Create some widgets to be placed inside
  btn = QtWidgets.QPushButton('press me')
  text = QtWidgets.QLineEdit('enter text')
  listWidget = QtWidgets.QListWidget()
  plot = pg.PlotWidget()
  
  # Create a grid layout to manage the widgets size and position
  layout = QtWidgets.QGridLayout()
  w.setLayout(layout)
  
  # Add widgets to the layout in their proper positions
  layout.addWidget(btn, 0, 0)  # button goes in upper-left
  layout.addWidget(text, 1, 0)  # text edit goes in middle-left
  layout.addWidget(listWidget, 2, 0)  # list widget goes in bottom-left
  layout.addWidget(plot, 0, 1, 3, 1)  # plot goes on right side, spanning 3 rows
  # Display the widget as a new window
  w.show()
  
  # Start the Qt event loop
  app.exec()  # or app.exec_() for PyQt5 / PySide2

More complex interfaces may be designed graphically using Qt Designer, which allows you
to simply drag widgets into your window to define its appearance.


Naming Conventions
------------------

Virtually every class in PyQtGraph is an extension of base classes provided by Qt. When
reading the documentation, remember that all of Qt's classes start with the letter 'Q',
whereas PyQtGraph's classes do not. When reading through the methods for any class, it
is often helpful to see which Qt base classes are used and look through the Qt
documentation as well.

Most of Qt's classes define signals which can be difficult to tell apart from regular
methods. Almost all signals explicitly defined by pyqtgraph are named beginning with
'sig' to indicate that these signals are not defined at the Qt level.

In most cases, classes which end in ``Widget`` are subclassed from :class:`QWidget` and
can therefore be used as a GUI element in a Qt window. Classes which end in ``Item`` are
subclasses of :class:`QGraphicsItem` and can only be displayed within a
:class:`QGraphicsView` instance, such as :class:`~pyqtgraph.GraphicsLayoutWidget` or
:class:`~pyqtgraph.PlotWidget`. 


Signals and Slots
--------------------------

For an overview of Qt's signal and slots mechanism, check out their `signals and slots`_
documentation.

When a :class:`Signal` is emitted, it triggers either other signals that it is connected
to, or a :class:`Slot`.  A slot is a method or stand-alone function that is run when a
signal that's connected to it is emitted.  


GraphicsView and GraphicsItems
------------------------------

PyQtGraph makes extensive usage of Qt's `Graphics View framework`_. This documentation
should be used as a reference if looking for a basis of PyQtGraph's inner workings.


Coordinate Systems and Transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

More information about the coordinate systems in Qt GraphicsView, read through the
`Graphics View Coordinate System`_ documentation.

To manipulate the shape and position of :class:`QGraphicsItem` objects,
:class:`QTransform` objects are applied. Sometimes we need to go "backwards" in a 
transformation, and while :class:`QTransform` provides an :meth:`QTransform.inverted` 
method, PyQtGraph avoids calling that object due to precision issues involving
`qFuzzyIsNull`_. Instead, when PyQtGraph needs to invert a :class:`QTransform`, it uses
:func:`~pyqtgraph.invertQTransform` which attempts to preserve the full precision.

It should be noted that many of the Qt GraphicsView methods use
:meth:`QTransform.inverted <QTransform.inverted>` internally, and there is nothing
PyQtGraph can do to avoid those calls.

Mouse and Keyboard Input
------------------------

Events Overview
^^^^^^^^^^^^^^^
Understanding events in a Qt application is fundamental before delving into specific
input handling such as mouse and keyboard:

- Events in Qt are conceptualized as user interactions with the application, each
  represented by an event object :class:`QEvent`.
- Various types of events correspond to different user interactions.
- Event objects encapsulate details concerning the specific occurrence.
- Dispatched to designated event handlers within the widget where the interaction
  occurs, these events allow for customizable responses.
- Handlers may be extended or redefined to modify widget response to interactions.

Mouse Events
^^^^^^^^^^^^
Interactions such as clicks, movements, and button releases are managed by overriding
methods including :meth:`QWidget.mousePressEvent`, :meth:`QWidget.mouseReleaseEvent`,
:meth:`QWidget.mouseDoubleClickEvent`, and :meth:`QWidget.mouseMoveEvent`.

Integration with PyQtGraph
~~~~~~~~~~~~~~~~~~~~~~~~~~
PyQtGraph utilizes :class:`QWidget` subclasses to present graphics and plots.
Consequently, the event-handling methods discussed can be directly integrated into
PyQtGraph widgets. This integration enables sophisticated interactive features in
applications that leverage PyQtGraph for visual data representation.

.. code-block:: python
  :caption: Handling Mouse Clicks in a PlotWidget

  from PyQt6.QtWidgets import QApplication, QMainWindow 
  from PyQt6.QtCore import Qt

  class MainWindow(QMainWindow):
      def __init__(self):
          super().__init__()
          # Set the Title of the window
          self.setWindowTitle('Mouse and Keyboard Event Demo')
          # Set the position and size of the window
          self.setGeometry(100, 100, 400, 300)

      # This method checks if the left mouse button was pressed on the widget 
      # and prints the position of the click.
      def mousePressEvent(self, event):
          if event.button() == Qt.MouseButton.LeftButton:
              print("Left mouse button pressed at:", event.position())

  # Initialize the QApplication
  app = QApplication([])
  window = MainWindow()
  window.show()
  app.exec()  # Start the event loop

This code snippet demonstrates initializing a basic PyQt6 application that responds to a
left mouse button click, illustrating the practical application of handling mouse events
in a PyQtGraph environment.


Keyboard Events
^^^^^^^^^^^^^^^
Keyboard inputs are similarly handled by overriding
:meth:`QWidget.keyPressEvent <QWidget.keyPressEvent>` and
:meth:`QWidget.keyReleaseEvent <QWidget.keyPressEvent>`, allowing applications to react
to various keystrokes and facilitate shortcuts and other interactions.


Integration with PyQtGraph
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
  :caption: Handling Keyboard Inputs

  from PyQt6.QtWidgets import QApplication, QMainWindow
  # Should work with PyQt5 / PySide2 / PySide6 as well
  from PyQt6.QtCore import Qt

  class MainWindow(QMainWindow):
  def __init__(self):
      super().__init__()
      # Set the Title of the window
      self.setWindowTitle('Keyboard Input Tracker')
      # Sets the position and size of the window
      self.setGeometry(100, 100, 400, 300)

  def keyPressEvent(self, event): # Checks if a specific key was pressed

      if event.key() == Qt.Key.Key_Escape:
          print("Escape key was pressed.")
      elif event.key() == Qt.Key.Key_Space:
          print("Space bar was pressed.")
      else:
          # The 'event.text()' method retrieves the character or character
          # associated with the key press, and then prints it to the console.
          print(f"Key pressed: {event.text()}")

  # Initialize the QApplication
  app = QApplication([])
  window = MainWindow()
  window.show()
  app.exec() # Starts the event loop


Event Propagation
^^^^^^^^^^^^^^^^^
In Qt, when an event is not handled by a widget, or the widget explicitly decides
against handling it, the event is propagated to its parent widget. This process,
commonly referred to as "bubbling", continues upward through the nested widgets until
the event is either handled or reaches the main window.

It is facilitated by methods such as :meth:`QEvent.accept <QEvent.accept>` and
:meth:`QEvent.ignore <QEvent.ignore>`, which allow developers to exert precise control
over the event flow.

.. code-block:: python
  :caption: Custom Event Handling

  class CustomButton(QPushButton):
      def mousePressEvent(self, event):
          # accept an event if it's caused by a right-click
          if event.button() == QtCore.Qt.MouseButton.RightButton:
              # The event is marked as handled, preventing further propagation
              event.accept()
          else:
              # Alternatively the event can be marked as unhandled, allowing it
              # to propagate further
              event.ignore()
  

QTimer
------

:class:`QTimer` is simply a Qt class that provides a high-level interface for creating
and managing timers in a Qt Application. This timers are used to perform an action
periodically. It can be used for tasks such as doing periodic data polling of the
tasks, updating the user interface, or triggering the events at regular intervals.

.. code-block:: python
  :caption: Example with QTimer

  import sys
  from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel
  from PyQt6.QtCore import QTimer, Qt

  class MainWindow(QMainWindow):
      def __init__(self):
          super().__init__()
          self.setWindowTitle("QTimer Example")
          self.setGeometry(120, 120, 450, 250)
          self.label = QLabel("Timer not begins", self)
          self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
          self.setCentralWidget(self.label)

          self.timer = QTimer(self)
          self.timer.timeout.connect(self.update_label)
          self.timer.start(1000)  # Timer set to 1 second (1000 ms)

      def update_label(self):
          self.label.setText(
            "Updated at: " + str(QTimer.remainingTime(self.timer))
          )

  if __name__ == "__main__":
      app = QApplication(sys.argv)
      window = MainWindow()
      window.show()
      sys.exit(app.exec())

Many of the PyQtGraph examples make use of :class:`QTimer`.

QThread
-------

The use of multithreading can help keep the GUI responsive while tasks that are taking a
long time to complete are running, blocking the completion of the Qt event loop.

Qt provides the :class:`QThread` class which allows for moving tasks to run on the
non-GUI thread. It can be extremely useful for tasks that are often waiting, such as
network communication, disk I/O, or any other tasks that runs independent of the main
GUI thread.

In the following example, the calculation for a prime number occurs in a separate
thread. While the calculation for the n-th prime number occurs, you will notice the GUI
does not freeze.

.. code-block:: python
  :caption: Demonstration of Using QThread To Perform Long Running Calculation

  # Example sourced from Rob Kent (@jazzycamel) and modified for PyQtGraph purposes
  # https://gist.github.com/jazzycamel/8abd37bf2d60cce6e01d 
  # SPDX-License-Identifier: MIT
  from itertools import count, islice
  import sys

  from PyQt6.QtCore import *
  from PyQt6.QtWidgets import *


  class Threaded(QObject):
      result=pyqtSignal(int)

      def __init__(self, parent=None, **kwargs):
          # intentionally not setting the parent
          super().__init__(parent=None, **kwargs)

      @pyqtSlot()
      def start(self):
          print("Thread started")

      @pyqtSlot(int)
      def calculatePrime(self, n):
          primes=(n for n in count(2) if all(n % d for d in range(2, n)))
          # sends the result across threads
          self.result.emit(list(islice(primes, 0, n))[-1])

  class Window(QWidget):
      requestPrime=pyqtSignal(int)

      def __init__(self, parent=None, **kwargs):
          super().__init__(parent, **kwargs)

          self._thread = QThread()
          # important to *not* set a parent, or .moveToThread will silently fail
          self._threaded = Threaded()
          self._threaded.result.connect(self.displayPrime)
          self.requestPrime.connect(self._threaded.calculatePrime)
          self._thread.started.connect(self._threaded.start)
          self._threaded.moveToThread(self._thread)

          qApp = QApplication.instance()
          if qApp is not None:
              qApp.aboutToQuit.connect(self._thread.quit)
          self._thread.start()

          layout = QVBoxLayout(self)
          self._iterationLineEdit = QLineEdit(
              self,
              placeholderText="Iteration (n)"
          )
          layout.addWidget(self._iterationLineEdit)
          self._requestButton = QPushButton(
              "Calculate Prime",
              self,
              clicked=self.primeRequested
          )
          layout.addWidget(self._requestButton)
          self._busy = QProgressBar(self)
          layout.addWidget(self._busy)
          self._resultLabel=QLabel("Result:", self)
          layout.addWidget(self._resultLabel)

      @pyqtSlot()
      def primeRequested(self):
          try:
              n = int(self._iterationLineEdit.text())
          except ValueError:
              # ignore input that can't be cast to int
              return
          self.requestPrime.emit(n)
          self._busy.setRange(0, 0)
          self._iterationLineEdit.setEnabled(False)
          self._requestButton.setEnabled(False)

      @pyqtSlot(int)
      def displayPrime(self, prime):
          self._resultLabel.setText(f"Result: {prime}")
          self._busy.setRange(0, 100)
          self._iterationLineEdit.setEnabled(True)
          self._requestButton.setEnabled(True)

  if __name__=="__main__":
      a = QApplication(sys.argv)
      g = Window()
      g.show()
      sys.exit(a.exec())


This example can be modified to handle a case where you want a thread to wait to collect
data from an external source, or other tasks that involve waiting, but where you do not
want the GUI to freeze.

.. _signals and slots: https://doc.qt.io/qt-6/signalsandslots.html
.. _Graphics View framework: https://doc.qt.io/qt-6/graphicsview.html
.. _Graphics View Coordinate System: 
  https://doc.qt.io/qt-6/graphicsview.html#the-graphics-view-coordinate-system
.. _qFuzzyIsNull: https://doc.qt.io/qt-6/qtnumeric.html#qFuzzyIsNull-1
