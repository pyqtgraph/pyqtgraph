"""
Script for invoking pytest with options to select Qt library
"""

import os, sys
import pytest

args = sys.argv[1:]
if '--pyside' in args:
    args.remove('--pyside')
    import PySide
elif '--pyqt4' in args:
    args.remove('--pyqt4')
    import PyQt4
elif '--pyqt5' in args:
    args.remove('--pyqt5')
    import PyQt5

import pyqtgraph as pg
pg.systemInfo()

os.exit(pytest.main(args))
    
    
