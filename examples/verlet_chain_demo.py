"""
Mechanical simulation of a chain using verlet integration.

Use the mouse to interact with one of the chains.

By default, this uses a slow, pure-python integrator to solve the chain link
positions. Unix users may compile a small math library to speed this up by
running the `examples/verlet_chain/make` script.

"""

import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

import verlet_chain

sim = verlet_chain.ChainSim()

if verlet_chain.relax.COMPILED:
    # Use more complex chain if compiled mad library is available.
    chlen1 = 80
    chlen2 = 60
    linklen = 1
else:
    chlen1 = 10
    chlen2 = 8
    linklen = 8

npts = chlen1 + chlen2

sim.mass = np.ones(npts)
sim.mass[int(chlen1 * 0.8)] = 100
sim.mass[chlen1-1] = 500
sim.mass[npts-1] = 200

sim.fixed = np.zeros(npts, dtype=bool)
sim.fixed[0] = True
sim.fixed[chlen1] = True

sim.pos = np.empty((npts, 2))
sim.pos[:chlen1, 0] = 0
sim.pos[chlen1:, 0] = 10
sim.pos[:chlen1, 1] = np.arange(chlen1) * linklen
sim.pos[chlen1:, 1] = np.arange(chlen2) * linklen
# to prevent miraculous balancing acts:
sim.pos += np.random.normal(size=sim.pos.shape, scale=1e-3)

links1 = [(j, i+j+1) for i in range(chlen1) for j in range(chlen1-i-1)]
links2 = [(j, i+j+1) for i in range(chlen2) for j in range(chlen2-i-1)]
sim.links = np.concatenate([np.array(links1), np.array(links2)+chlen1, np.array([[chlen1-1, npts-1]])])

p1 = sim.pos[sim.links[:,0]]
p2 = sim.pos[sim.links[:,1]]
dif = p2-p1
sim.lengths = (dif**2).sum(axis=1) ** 0.5
sim.lengths[(chlen1-1):len(links1)] *= 1.05  # let auxiliary links stretch a little
sim.lengths[(len(links1)+chlen2-1):] *= 1.05
sim.lengths[-1] = 7

push1 = np.ones(len(links1), dtype=bool)
push1[chlen1:] = False
push2 = np.ones(len(links2), dtype=bool)
push2[chlen2:] = False
sim.push = np.concatenate([push1, push2, np.array([True], dtype=bool)])

sim.pull = np.ones(sim.links.shape[0], dtype=bool)
sim.pull[-1] = False

# move chain initially just to generate some motion if the mouse is not over the window
mousepos = np.array([30, 20])


def display():
    global view, sim
    view.clear()
    view.addItem(sim.makeGraph())

def relaxed():
    global app
    display()
    app.processEvents()

def mouse(pos):
    global mousepos
    pos = view.mapSceneToView(pos)
    mousepos = np.array([pos.x(), pos.y()])

def update():
    global mousepos
    #sim.pos[0] = sim.pos[0] * 0.9 + mousepos * 0.1
    s = 0.9
    sim.pos[0] = sim.pos[0] * s + mousepos * (1.0-s)
    sim.update()

app = pg.mkQApp()
win = pg.GraphicsLayoutWidget()
win.show()
view = win.addViewBox()
view.setAspectLocked(True)
view.setXRange(-100, 100)
#view.autoRange()

view.scene().sigMouseMoved.connect(mouse)

#display()
#app.processEvents()

sim.relaxed.connect(relaxed)
sim.init()
sim.relaxed.disconnect(relaxed)

sim.stepped.connect(display)

timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(16)


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
