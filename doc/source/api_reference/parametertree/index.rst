.. _parametertree:

Parameter Trees
===============

.. currentmodule:: pyqtgraph.parametertree

Parameter trees are a system for handling hierarchies of parameters while automatically generating one or more GUIs to display and interact with the parameters.
This feature is commonly seen, for example, in user interface design applications which display a list of editable properties for each widget.
Parameters generally have a name, a data type (int, float, string, color, etc), and a value matching the data type. Parameters may be grouped and nested to form hierarchies and may be subclassed to provide custom behavior and display widgets.

PyQtGraph's parameter tree system works similarly to the model-view architecture used by some components of Qt:

- A :class:`Parameter` is a purely data-handling class that exists independent of any graphical interface.
- A :class:`ParameterItem` is an interactive graphical representation of a :class:`Parameter`.
- A :class:`ParameterTree` is a widget that automatically generates a graphical interface which represents the state of a hierarchy of Parameter objects and allows the user to edit the values within that hierarchy.

This separation of data (model) and graphical interface (view) allows the same data to be represented multiple times and in a variety of different ways.
For example, a floating point number parameter could be represented by a slider or a spinbox, or both.

For more information, see the 'parametertree' example included with pyqtgraph and the API reference:

.. toctree::
    :maxdepth: 2

    apiref
