#!/usr/bin/env python



import os
from string import Template

## Root path of project
ROOT = os.path.abspath(os.path.join( os.path.dirname(os.path.abspath(__file__)), "../"))
print ROOT

###########################################################################
## Generate the examples
###########################################################################

## =================================
# Template for an example
# nb Including this .. automodule:: examples.$fname makes sphinx run all the examples ;-( DONT DO
tplExample = Template("""
.. _ex$fname:

$fname
==============================================

.. literalinclude:: ../../../examples/$fname.py
   :language: python
   :linenos:
""")



## Nuke all the files in example/
os.popen('rm ' + ROOT + "/doc/source/examples/*.rst")

# get list of source files
example_source_files =  sorted(os.listdir(ROOT + "/examples"))

# index list of examples for spooling to examples/index.rst
index = []

### Write out each example ###
for f in example_source_files:
    
    fname, ext = os.path.splitext(os.path.basename(f))
    #print fname, ext
    
    ## we only deal with .py files and not __init
    if ext == ".py" and fname[0:2] != "__":
        rst_file = open(ROOT + "/doc/source/examples/%s.rst" % fname.lower(), "w")
        txt = tplExample.substitute(fname=fname)
        rst_file.write(txt)
        rst_file.close()
        
        index.append(fname.lower())

### Wwrite out index ###

## the examples/index.rst contents
idx_string = """
.. _examples:

Examples
================
PyQtGraph includes an extensive set in the **examples/** directory listed below.

Run Launcher
-----------------

.. code-block:: python

    # In a script file
    import pyqtgraph.examples
    pyqtgraph.examples.run()


.. code-block:: bash

    # On command line
    python -m pyqtgraph.examples


List of examples
-----------------

.. toctree::
    :maxdepth: 2
    
"""

for f in index:
    idx_string += "    %s\n" % f
    
index_file = open(ROOT + "/doc/source/examples/index.rst", "w") 
index_file.write(idx_string)
index_file.close()

print "examples done :-)"



