#!/usr/bin/python
"""
Script for compiling Qt Designer .ui files to .py
"""


import os
import subprocess
import sys

pyqt6uic = 'pyuic6'

usage = """Compile .ui files to .py for PyQt6

  Usage: python rebuildUi.py [--force] [.ui files|search paths]

  May specify a list of .ui files and/or directories to search recursively for .ui files.
"""

args = sys.argv[1:]

if '--force' in args:
    force = True
    args.remove('--force')
else:
    force = False

if len(args) == 0:
    print(usage)
    sys.exit(-1)

uifiles = []
for arg in args:
    if os.path.isfile(arg) and arg.endswith('.ui'):
        uifiles.append(arg)
    elif os.path.isdir(arg):
        # recursively search for ui files in this directory
        for path, sd, files in os.walk(arg):
            uifiles.extend(os.path.join(path, f) for f in files if f.endswith('.ui'))
    else:
        print('Argument "%s" is not a directory or .ui file.' % arg)
        sys.exit(-1)

compiler = pyqt6uic
extension = '_generic.py'
# rebuild all requested ui files
for ui in uifiles:
    base, _ = os.path.splitext(ui)
    py = base + ext
    if not force and os.path.exists(py) and os.stat(ui).st_mtime <= os.stat(py).st_mtime:
        print(f"Skipping {py}; already compiled.")
    else:
        cmd = f'{compiler} {ui} > {py}'
        print(cmd)
        try:
            subprocess.check_call(cmd, shell=True)
        except subprocess.CalledProcessError:
            os.remove(py)
        else:
            print(f"{py} created, modify import to import from pyqtgraph.Qt not PyQt6")
