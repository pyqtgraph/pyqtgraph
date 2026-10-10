import os
import sys

import pytest

from pyqtgraph.Qt import QtCore, mkQApp

try:
    QtCore.QLoggingCategory.setFilterRules("qt.pyside.libpyside.warning=true")
except AttributeError:
    pass  # PySide2 does not expose `QLoggingCategory` although Qt5 has it


@pytest.fixture
def tmp_module(tmp_path):
    module_path = os.fsdecode(tmp_path)
    sys.path.insert(0, module_path)
    yield module_path
    sys.path.remove(module_path)


@pytest.fixture
def default_locale():
    """Set the default locale in a test and restore the previous one afterwards.

    Yields ``setDefault(locale, applicationWide=False)``. With
    ``applicationWide=True`` it also sends the application a
    ``LocaleChange`` event, as a system locale change does, so widgets
    without a locale of their own pick up the new default.
    """
    app = mkQApp()
    previous = QtCore.QLocale()

    def setDefault(locale, applicationWide=False):
        QtCore.QLocale.setDefault(locale)
        if applicationWide:
            QtCore.QCoreApplication.sendEvent(
                app, QtCore.QEvent(QtCore.QEvent.Type.LocaleChange))

    yield setDefault
    QtCore.QLocale.setDefault(previous)
    QtCore.QCoreApplication.sendEvent(
        app, QtCore.QEvent(QtCore.QEvent.Type.LocaleChange))
