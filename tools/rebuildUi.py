#!/usr/bin/python
"""
Script for compiling Qt Designer .ui files to .py



"""
import os, sys, subprocess, tempfile

pyqtuic = 'pyuic4'
pysideuic = 'pyside-uic'
pyside2uic = 'pyside2-uic'
pyqt5uic = 'pyuic5'

usage = """Compile .ui files to .py for all supported pyqt/pyside versions.

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
            for f in files:
                if not f.endswith('.ui'):
                    continue
                uifiles.append(os.path.join(path, f))
    else:
        print('Argument "%s" is not a directory or .ui file.' % arg)
        sys.exit(-1)

# rebuild all requested ui files
for ui in uifiles:
    base, _ = os.path.splitext(ui)
    for compiler, ext in [(pyqtuic, '_pyqt.py'), (pysideuic, '_pyside.py'), (pyqt5uic, '_pyqt5.py'), (pyside2uic, '_pyside2.py')]:
        py = base + ext
        if not force and os.path.exists(py) and os.stat(ui).st_mtime <= os.stat(py).st_mtime:
            print("Skipping %s; already compiled." % py)
        else:
            cmd = '%s %s > %s' % (compiler, ui, py)
            print(cmd)
            try:
                subprocess.check_call(cmd, shell=True)
            except subprocess.CalledProcessError:
                os.remove(py)
