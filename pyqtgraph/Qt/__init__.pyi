"""
This stub file is to aid in the PyCharm and VSCode auto-completion of the Qt imports.
"""

from . import QtCore as QtCore
from . import QtGui as QtGui
from . import QtSvg as QtSvg
from . import QtTest as QtTest
from . import QtWidgets as QtWidgets

App: QtWidgets.QApplication
VERSION_INFO: str
QT_LIB: str
QtVersion: str

def exec_() -> QtWidgets.QApplication: ...
def mkQApp(name: str | None = None) -> QtWidgets.QApplication: ...
def isQObjectAlive(obj: QtCore.QObject) -> bool: ...
