# -*- coding: utf-8 -*-
#import sip
#sip.setapi('QString', 1)

import pyqtgraph as pg
pg.mkQApp()

import pyqtgraph.dockarea as da

def test_dock():
    name = pg.asUnicode("évènts_zàhéér")
    dock = da.Dock(name=name)
    # make sure unicode names work correctly
    assert dock.name() == name
    # no surprises in return type.
    assert type(dock.name()) == type(name)

def test_closable_dock():
    name = "Test close dock"
    dock = da.Dock(name=name, closable=True)

    assert dock.label.closeButton != None
