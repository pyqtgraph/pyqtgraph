Rapid GUI prototyping
=====================

[Just an overview; documentation is not complete yet]

PyQtGraph offers several powerful features which are commonly used in engineering and scientific applications.

Parameter Trees
---------------

The parameter tree system provides a widget displaying a tree of modifiable values similar to those used in most GUI editor applications. This allows a large number of variables to be controlled by the user with relatively little programming effort. The system also provides separation between the data being controlled and the user interface controlling it (model/view architecture). Parameters may be grouped/nested to any depth and custom parameter types can be built by subclassing from Parameter and ParameterItem.

See the `parametertree documentation <parametertree>`_ for more information.


Visual Programming Flowcharts
-----------------------------

PyQtGraph's flowcharts provide a visual programming environment similar in concept to LabView--functional modules are added to a flowchart and connected by wires to define a more complex and arbitrarily configurable algorithm. A small number of predefined modules (called Nodes) are included with pyqtgraph, but most flowchart developers will want to define their own library of Nodes. At their core, the Nodes are little more than 1) a Python function 2) a list of input/output terminals, and 3) an optional widget providing a control panel for the Node. Nodes may transmit/receive any type of Python object via their terminals.

See the `flowchart documentation <flowchart>`_ and the flowchart examples for more information.


Graphical Canvas
----------------

The Canvas is a system designed to allow the user to add/remove items to a 2D canvas similar to most vector graphics applications. Items can be translated/scaled/rotated and each item may define its own custom control interface.


Dockable Widgets
----------------

The dockarea system allows the design of user interfaces which can be rearranged by the user at runtime. Docks can be moved, resized, stacked, and torn out of the main window. This is similar in principle to the docking system built into Qt, but offers a more deterministic dock placement API (in Qt it is very difficult to programatically generate complex dock arrangements). Additionally, Qt's docks are designed to be used as small panels around the outer edge of a window. PyQtGraph's docks were created with the notion that the entire window (or any portion of it) would consist of dockable components.


