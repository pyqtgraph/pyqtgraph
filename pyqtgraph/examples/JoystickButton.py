"""
JoystickButton is a button with x/y values. When the button is depressed and the
mouse dragged, the x/y values change to follow the mouse.
When the mouse button is released, the x/y values change to 0,0 (rather like 
letting go of the joystick).
"""

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

app = pg.mkQApp("Joystick Button Example")
mw = QtWidgets.QMainWindow()
mw.resize(300,50)
mw.setWindowTitle('pyqtgraph example: JoystickButton')
cw = QtWidgets.QWidget()
mw.setCentralWidget(cw)
layout = QtWidgets.QGridLayout()
cw.setLayout(layout)
mw.show()

l1 = pg.ValueLabel(siPrefix=True, suffix='m')
l2 = pg.ValueLabel(siPrefix=True, suffix='m')
jb = pg.JoystickButton()
jb.setFixedWidth(30)
jb.setFixedHeight(30)


layout.addWidget(l1, 0, 0)
layout.addWidget(l2, 0, 1)
layout.addWidget(jb, 0, 2)

x = 0
y = 0
def update():
    global x, y, l1, l2, jb
    dx, dy = jb.getState()
    x += dx * 1e-3
    y += dy * 1e-3
    l1.setValue(x)
    l2.setValue(y)
    # If the timer frequency is fast enough for the Qt platform (in case
    # the frequency is increased or if the desktop is overloaded), the GUI
    # might get stuck because the event loop won't manage to respond to
    # events such as window resize etc while the timer is running. This
    # forces the timer to process the GUI events and to provide a smooth
    # experience. 
    app.processEvents()
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(30)
    
if __name__ == '__main__':
    pg.exec()
