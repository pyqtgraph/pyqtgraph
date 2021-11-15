import pytest

import pyqtgraph as pg

pg.mkQApp()

import pyqtgraph.dockarea as da


def test_dock():
    name = "évènts_zàhéér"
    dock = da.Dock(name=name)
    # make sure unicode names work correctly
    assert dock.name() == name
    # no surprises in return type.
    assert type(dock.name()) == type(name)

def test_closable_dock():
    name = "Test close dock"
    dock = da.Dock(name=name, closable=True)

    assert dock.label.closeButton is not None

def test_hide_title_dock():
    name = "Test hide title dock"
    dock = da.Dock(name=name, hideTitle=True)

    assert dock.labelHidden == True

def test_close():
    name = "Test close dock"
    dock = da.Dock(name=name, hideTitle=True)
    with pytest.warns(Warning):
        dock.close()
