"""
Demonstrates use of PlotWidget class. This is little more than a 
GraphicsView with a PlotItem placed in its center.
"""

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

app = pg.mkQApp()
mw = QtWidgets.QMainWindow()
mw.setWindowTitle('pyqtgraph example: PlotWidget')
mw.resize(800,800)
cw = QtWidgets.QWidget()
mw.setCentralWidget(cw)
l = QtWidgets.QVBoxLayout()
cw.setLayout(l)

pw = pg.PlotWidget(name='Plot1')  ## giving the plots names allows us to link their axes together
l.addWidget(pw)
pw2 = pg.PlotWidget(name='Plot2')
l.addWidget(pw2)
pw3 = pg.PlotWidget(name='Plot3')
l.addWidget(pw3)

mw.show()

## Create an empty plot curve to be filled later, set its pen
p1 = pw.plot()
p1.setPen((200,200,100))

## Add in some extra graphics
rect = QtWidgets.QGraphicsRectItem(QtCore.QRectF(0, 0, 1, 5e-11))
rect.setPen(pg.mkPen(100, 200, 100))
pw.addItem(rect)

pw.setLabel('left', 'Value', units='V')
pw.setLabel('bottom', 'Time', units='s')
pw.setXRange(0, 2)
pw.setYRange(0, 1e-10)

def rand(n):
    data = np.random.random(n)
    data[int(n*0.1):int(n*0.13)] += .5
    data[int(n*0.18)] += 2
    data[int(n*0.1):int(n*0.13)] *= 5
    data[int(n*0.18)] *= 20
    data *= 1e-12
    return data, np.arange(n, n+len(data)) / float(n)
    

def updateData():
    yd, xd = rand(10000)
    p1.setData(y=yd, x=xd)

## Start a timer to rapidly update the plot in pw
t = QtCore.QTimer()
t.timeout.connect(updateData)
t.start(50)
#updateData()

## Multiple parameterized plots--we can autogenerate averages for these.
for i in range(0, 5):
    for j in range(0, 3):
        yd, xd = rand(10000)
        pw2.plot(y=yd*(j+1), x=xd, params={'iter': i, 'val': j})

## Demonstrate multi-button click support with three different examples on pw3
pw3.setTitle("Multi-button Click Examples")

n = 100  # Same length for all plots

## Example 1: Scatter plot only with left/right buttons
scatter1 = pw3.plot(
    x=np.arange(n),
    y=np.random.normal(size=n) * 0.5 + 5,
    pen=None,
    symbol='o',
    symbolSize=8,
    symbolBrush=(100, 100, 255, 200),
    name='Scatter (L/R)',
    clickable=True,
    clickButtons=QtCore.Qt.MouseButton.LeftButton | QtCore.Qt.MouseButton.RightButton
)

def scatter1_clicked(item, ev):
    if ev.button() == QtCore.Qt.MouseButton.LeftButton:
        print("Scatter 1: Left button clicked")
    elif ev.button() == QtCore.Qt.MouseButton.RightButton:
        print("Scatter 1: Right button clicked")

def scatter1_points_clicked(item, points, ev):
    button_name = "Left" if ev.button() == QtCore.Qt.MouseButton.LeftButton else "Right"
    print(f"Scatter 1: {button_name} clicked on {len(points)} point(s) at x={points[0].pos().x():.2f}")

scatter1.sigClicked.connect(scatter1_clicked)
scatter1.sigPointsClicked.connect(scatter1_points_clicked)

## Example 2: Curve only with left button
curve_only = pw3.plot(
    x=np.arange(n),
    y=np.sin(np.arange(n) * 0.1),
    pen=pg.mkPen('w', width=2),
    name='Curve (L only)',
    clickable=True,
    clickButtons=QtCore.Qt.MouseButton.LeftButton
)
curve_only.setShadowPen(pg.mkPen((70,70,30), width=6, cosmetic=True))

def curve_clicked(item, ev):
    print(f"Curve: Left clicked at x={ev.pos().x():.2f}")

curve_only.sigClicked.connect(curve_clicked)

## Example 3: Combined scatter + curve with left/right/middle buttons
combined = pw3.plot(
    x=np.arange(n),
    y=np.cos(np.arange(n) * 0.15) - 3 + np.random.normal(size=n) * 0.2,
    pen=pg.mkPen('c', width=2),
    symbol='t',
    symbolSize=10,
    symbolBrush=(255, 100, 100, 150),
    name='Combined (L/R/M)',
    clickable=True,
    clickButtons=QtCore.Qt.MouseButton.LeftButton |
                 QtCore.Qt.MouseButton.RightButton |
                 QtCore.Qt.MouseButton.MiddleButton
)

def combined_clicked(item, ev):
    button_name = {
        QtCore.Qt.MouseButton.LeftButton: "Left",
        QtCore.Qt.MouseButton.RightButton: "Right",
        QtCore.Qt.MouseButton.MiddleButton: "Middle"
    }.get(ev.button(), "Unknown")
    print(f"Combined: {button_name} button clicked (curve or scatter)")

def combined_points_clicked(item, points, ev):
    button_name = {
        QtCore.Qt.MouseButton.LeftButton: "Left",
        QtCore.Qt.MouseButton.RightButton: "Right",
        QtCore.Qt.MouseButton.MiddleButton: "Middle"
    }.get(ev.button(), "Unknown")
    print(f"Combined: {button_name} clicked on scatter point at x={points[0].pos().x():.2f}, y={points[0].pos().y():.2f}")

combined.sigClicked.connect(combined_clicked)
combined.sigPointsClicked.connect(combined_points_clicked)

pw3.addLegend()
pw3.setXRange(-5, 105)
pw3.setYRange(-5, 7)

lr = pg.LinearRegionItem([30, 70], bounds=[0, 100], movable=True)
pw3.addItem(lr)
line = pg.InfiniteLine(angle=90, movable=True, pos=50)
pw3.addItem(line)
line.setBounds([0, 100])

if __name__ == '__main__':
    pg.exec()
