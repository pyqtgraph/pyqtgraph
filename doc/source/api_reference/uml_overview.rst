:html_theme.sidebar_secondary.remove:

UML class diagram
=================

.. _uml_diagram:

The UML class diagram below gives an overview of the most important classes and their relations.

The green boxes represent Qt classes, the purple boxes are PyQtGraph classes.

The black arrows indicate inheritance between two classes (with the parent class always above the child classes.)

The gray lines with the diamonds indicate an aggregation relation. For example the :class:`PlotDataItem <pyqtgraph.PlotDataItem>` class has a ``curve`` attribute that is a reference to a :class:`PlotCurveItem <pyqtgraph.PlotCurveItem>` object.


.. If it's stupid, and it works, it's not stupid
.. Inlining SVG code, not using <img> tags so nodes can act as links and be clicked

.. raw:: html

	<div class="only-dark">

.. raw:: html
	:file: ../images/overview_uml-dark_mode.svg

.. raw:: html

	</div><div class="only-light">

.. raw:: html
	:file: ../images/overview_uml-light_mode.svg

.. raw:: html

	</div>

.. end of not stupid stupidity
