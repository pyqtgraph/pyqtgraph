# -*- coding: utf-8 -*-
"""
Using ProgressDialog to show progress updates in a nested process.

"""
import initExample ## Add path to library (just for examples; you do not need this)

import time
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

app = pg.mkQApp("Process Dialog Example")


def runStage(i):
    """Waste time for 2 seconds while incrementing a progress bar.
    """
    with pg.ProgressDialog("Running stage %s.." % i, maximum=100, nested=True) as dlg:
        for j in range(100):
            time.sleep(0.02)
            dlg += 1
            if dlg.wasCanceled():
                print("Canceled stage %s" % i)
                break


def runManyStages(i):
    """Iterate over runStage() 3 times while incrementing a progress bar.
    """
    with pg.ProgressDialog("Running stage %s.." % i, maximum=3, nested=True, wait=0) as dlg:
        for j in range(1,4):
            runStage('%d.%d' % (i, j))
            dlg += 1
            if dlg.wasCanceled():
                print("Canceled stage %s" % i)
                break


with pg.ProgressDialog("Doing a multi-stage process..", maximum=5, nested=True, wait=0) as dlg1:
    for i in range(1,6):
        if i == 3:
            # this stage will have 3 nested progress bars
            runManyStages(i)
        else:
            # this stage will have 2 nested progress bars
            runStage(i)
        
        dlg1 += 1
        if dlg1.wasCanceled():
            print("Canceled process")
            break


