Visual Programming with Flowcharts
==================================

PyQtGraph's flowcharts provide a visual programming environment similar in concept to LabView--functional modules are added to a flowchart and connected by wires to define a more complex and arbitrarily configurable algorithm. A small number of predefined modules (called Nodes) are included with pyqtgraph, but most flowchart developers will want to define their own library of Nodes. At their core, the Nodes are little more than 1) a python function 2) a list of input/output terminals, and 3) an optional widget providing a control panel for the Node. Nodes may transmit/receive any type of Python object via their terminals.

One major limitation of flowcharts is that there is no mechanism for looping within a flowchart. (however individual Nodes may contain loops (they may contain any Python code at all), and an entire flowchart may be executed from within a loop). 

There are two distinct modes of executing the code in a flowchart:
    
1. Provide data to the input terminals of the flowchart. This method is slower and will provide a graphical representation of the data as it passes through the flowchart. This is useful for debugging as it allows the user to inspect the data at each terminal and see where exceptions occurred within the flowchart.
2. Call :func:`Flowchart.process() <pyqtgraph.flowchart.Flowchart.process>`. This method does not update the displayed state of the flowchart and only retains the state of each terminal as long as it is needed. Additionally, Nodes which do not contribute to the output values of the flowchart (such as plotting nodes) are ignored. This mode allows for faster processing of large data sets and avoids memory issues which can occur if too much data is present in the flowchart at once (e.g., when processing image data through several stages). 

See the flowchart example for more information.

API Reference:

.. toctree::
    :maxdepth: 2

    flowchart
    node
    terminal

Basic Use
---------

Flowcharts are most useful in situations where you have a processing stage in your application that you would like to be arbitrarily configurable by the user. Rather than giving a pre-defined algorithm with parameters for the user to tweak, you supply a set of pre-defined functions and allow the user to arrange and connect these functions how they like. A very common example is the use of filter networks in audio / video processing applications.

To begin, you must decide what the input and output variables will be for your flowchart. Create a flowchart with one terminal defined for each variable::
    
    ## This example creates just a single input and a single output.
    ## Flowcharts may define any number of terminals, though.
    from pyqtgraph.flowchart import Flowchart
    fc = Flowchart(terminals={
        'nameOfInputTerminal': {'io': 'in'},
        'nameOfOutputTerminal': {'io': 'out'}    
    })
    
In the example above, each terminal is defined by a dictionary of options which define the behavior of that terminal (see :func:`Terminal.__init__() <pyqtgraph.flowchart.Terminal.__init__>` for more information and options). Note that Terminals are not typed; any python object may be passed from one Terminal to another.

Once the flowchart is created, add its control widget to your application::
    
    ctrl = fc.ctrlWidget()
    myLayout.addWidget(ctrl)  ## read Qt docs on QWidget and layouts for more information

The control widget provides several features:
    
* Displays a list of all nodes in the flowchart containing the control widget for
  each node.
* Provides access to the flowchart design window via the 'flowchart' button
* Interface for saving / restoring flowcharts to disk.

At this point your user has the ability to generate flowcharts based on the built-in node library. It is recommended to provide a default set of flowcharts for your users to build from.

All that remains is to process data through the flowchart. As noted above, there are two ways to do this:

.. _processing methods:

1. Set the values of input terminals with :func:`Flowchart.setInput() <pyqtgraph.flowchart.Flowchart.setInput>`, then read the values of output terminals with :func:`Flowchart.output() <pyqtgraph.flowchart.Flowchart.output>`::
    
       fc.setInput(nameOfInputTerminal=newValue)
       output = fc.output()  # returns {terminalName:value}
       
   This method updates all of the values displayed in the flowchart design window, allowing the user to inspect values at all terminals in the flowchart and indicating the location of errors that occurred during processing.
2. Call :func:`Flowchart.process() <pyqtgraph.flowchart.Flowchart.process>`::
    
       output = fc.process(nameOfInputTerminal=newValue)
       
   This method processes data without updating any of the displayed terminal values. Additionally, all :func:`Node.process() <pyqtgraph.flowchart.Node.process>` methods are called with display=False to request that they not invoke any custom display code. This allows data to be processed both more quickly and with a smaller memory footprint, but errors that occur during Flowchart.process() will be more difficult for the user to diagnose. It is thus recommended to use this method for batch processing through flowcharts that have already been tested and debugged with method 1.

Implementing Custom Nodes
-------------------------

PyQtGraph includes a small library of built-in flowchart nodes. This library is intended to cover some of the most commonly-used functions as well as provide examples for some more exotic Node types. Most applications that use the flowchart system will find the built-in library insufficient and will thus need to implement custom Node classes. 

A node subclass implements at least:
    
1) A list of input / output terminals and their properties
2) A :func:`process() <pyqtgraph.flowchart.Node.process>` function which takes the names of input terminals as keyword arguments and returns a dict with the names of output terminals as keys.

Optionally, a Node subclass can implement the :func:`ctrlWidget() <pyqtgraph.flowchart.Node.ctrlWidget>` method, which must return a QWidget (usually containing other widgets) that will be displayed in the flowchart control panel. A minimal Node subclass looks like::
    
    class SpecialFunctionNode(Node):
        """SpecialFunction: short description
        
        This description will appear in the flowchart design window when the user 
        selects a node of this type.
        """
        nodeName = 'SpecialFunction' # Node type name that will appear to the user.
         
        def __init__(self, name):  # all Nodes are provided a unique name when they
                                   # are created.
            Node.__init__(self, name, terminals={  # Initialize with a dict 
                                                   # describing the I/O terminals
                                                   # on this Node.
                'inputTerminalName': {'io': 'in'},
                'anotherInputTerminal': {'io': 'in'},
                'outputTerminalName': {'io': 'out'},
                })
                
        def process(self, **kwds):
            # kwds will have one keyword argument per input terminal.
            
            return {'outputTerminalName': result}
        
        def ctrlWidget(self):  # this method is optional
            return someQWidget

Some nodes implement fairly complex control widgets, but most nodes follow a simple form-like pattern: a list of parameter names and a single value (represented as spin box, check box, etc..) for each parameter. To make this easier, the :class:`~pyqtgraph.flowchart.library.common.CtrlNode` subclass allows you to instead define a simple data structure that CtrlNode will use to automatically generate the control widget. This is used in  many of the built-in library nodes (especially the filters).

There are many other optional parameters for nodes and terminals -- whether the user is allowed to add/remove/rename terminals, whether one terminal may be connected to many others or just one, etc. See the documentation on the :class:`~pyqtgraph.flowchart.Node` and :class:`~pyqtgraph.flowchart.Terminal` classes for more details.

After implementing a new Node subclass, you will most likely want to register the class so that it appears in the menu of Nodes the user can select from::
    
    import pyqtgraph.flowchart.library as fclib
    fclib.registerNodeType(SpecialFunctionNode, [('Category', 'Sub-Category')])
    
The second argument to registerNodeType is a list of tuples, with each tuple describing a menu location in which SpecialFunctionNode should appear.
    
See the FlowchartCustomNode example for more information.


Debugging Custom Nodes
^^^^^^^^^^^^^^^^^^^^^^

When designing flowcharts or custom Nodes, it is important to set the input of the flowchart with data that at least has the same types and structure as the data you intend to process (see `processing methods`_ #1 above). When you use :func:`Flowchart.setInput() <pyqtgraph.flowchart.Flowchart.setInput>`, the flowchart displays visual feedback in its design window that can tell you what data is present at any terminal and whether there were errors in processing. Nodes that generated errors are displayed with a red border. If you select a Node, its input and output values will be displayed as well as the exception that occurred while the node was processing, if any.


Using Nodes Without Flowcharts
------------------------------

Flowchart Nodes implement a very useful generalization in data processing by combining a function with a GUI for configuring that function. This generalization is useful even outside the context of a flowchart. For example::
    
    ## We defined a useful filter Node for use in flowcharts, but would like to 
    ## re-use its processing code and GUI without having a flowchart present.
    filterNode = MyFilterNode("filterNodeName")
    
    ## get the Node's control widget and place it inside the main window
    filterCtrl = filterNode.ctrlWidget()
    someLayout.addWidget(filterCtrl)
    
    ## later on, process data through the node
    filteredData = filterNode.process(inputTerminal=rawData)


