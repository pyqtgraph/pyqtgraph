XML String Serialization
========================

While Parameter and ParameterTree objects can be used for many tasks,
saving/restoring their state is so far only done using a nested dictionary
structure, see :meth:`pyqtgraph.parametertree.Parameter.saveState` and
:meth:`pyqtgraph.parametertree.Parameter.restoreState`.
But when you need to save a Parameter state
in a file for later use you need serialization. Parameter may store builtins python
object whose serialization can be easily done using JSON or XML serializer.
But for more exotic ones, that may prove difficult. Another goal is to make the serialization
into a human readable form.

For these reasons, a XML Serializer has been built on top of the Parameter structure.
This is handled through the use of a factory (see Factory Pattern for more information)
called :class:`pyqtgraph.parametertree.XMLParameterFactory`.
Given a Parameter (in particular its type), the factory will be able to generate a XML structure,
easily dumped into a binary string hence serialization. For this to work each Parameter
object (and especially subtypes) should implement an interface to translate from
Parameter to XML string and back.


For the serialization to happen, each subclassed Parameter object must implement a dedicated interface.
This one is "forced" upon the Parameter class using a MixIn object:
the :class:`pyqtgraph.parametertree.XMLParameter`. This object implements methods to serialize back and forth
options shared between all Parameter object. It declares also two abstract method (that should therefore be
implemented in child Parameter) to translate into serializable dictionaries specific options such as the
**value** but not limited to it.

Parameters actually taken into account are:

* GroupParameter
* SimpleParameter (excluding the colormap subtype)

.. autoclass:: pyqtgraph.parametertree.XMLParameterFactory
    :members: parameter_to_xml_string, xml_string_to_parameter



.. autoclass:: pyqtgraph.parametertree.XMLParameter
    :members:




