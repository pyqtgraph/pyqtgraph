"""
Script for invoking pytest with options to select Qt library
"""

import sys
import pytest

args = sys.argv[1:]
if '--pyqt5' in args:
    args.remove('--pyqt5')
    import PyQt5
elif '--pyside2' in args:
    args.remove('--pyside2')
    import PySide2
elif '--pyside6' in args:
    args.remove('--pyside6')
    import PySide6
elif '--pyqt6' in args:
    args.remove('--pyqt6')
    import PyQt6

import pyqtgraph as pg
pg.systemInfo()
pytest.main(args)
