#!/usr/bin/env python

## Generate the examples

import os
from string import Template

## Includeing this .. automodule:: examples.$fname makes sphinx run all the examples ;-(

tplExample = Template("""
.. _ex$fname:

$fname
==============================================

.. literalinclude:: ../../../examples/$fname.py
   :language: python
   :linenos:
""")

ROOT = os.path.abspath(os.path.join( os.path.dirname(os.path.abspath(__file__)), "../"))
print ROOT

## first nuke all the files in example/
os.popen('rm ' + ROOT + "/doc/source/example/*.rst")

example_source_files =  sorted(os.listdir(ROOT + "/examples"))
index = []

for f in example_source_files:
    
    fname, ext = os.path.splitext(os.path.basename(f))
    #print fname, ext
    if ext == ".py" and fname[0:2] != "__":
        rst_file = open(ROOT + "/doc/source/example/%s.rst" % fname.lower(), "w")
        txt = tplExample.substitute(fname=fname)
        rst_file.write(txt)
        rst_file.close()
        
        index.append(fname.lower())

idx_string = """Examples
================
Below is a summary of files in the **examples/** directory

.. toctree::
    :maxdepth: 2
    
"""

for f in index:
    idx_string += "    %s\n" % f
    
index_file = open(ROOT + "/doc/source/example/index.rst", "w") 
index_file.write(idx_string)
index_file.close()


