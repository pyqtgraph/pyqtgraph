import pyqtgraph as pg
from pyqtgraph.Qt import QtTest, QtGui, QtCore
from pyqtgraph.tests import mouseDrag
pg.mkQApp()

qWait = QtTest.QTest.qWait


def test_mouseInteraction():
    plt = pg.plot()
    plt.scene().minDragTime = 0  # let us simulate mouse drags very quickly.
    vline = plt.addLine(x=0, movable=True)
    plt.addItem(vline)
    hline = plt.addLine(y=0, movable=True)
    plt.setXRange(-10, 10)
    plt.setYRange(-10, 10)
    
    # test horizontal drag
    pos = plt.plotItem.vb.mapViewToScene(pg.Point(0,5)).toPoint()
    pos2 = pos - QtCore.QPoint(200, 200)
    mouseDrag(plt, pos, pos2, QtCore.Qt.LeftButton)
    px = vline.pixelLength(pg.Point(1, 0), ortho=True)
    assert abs(vline.value() - plt.plotItem.vb.mapSceneToView(pos2).x()) <= px

    # test missed drag
    pos = plt.plotItem.vb.mapViewToScene(pg.Point(5,0)).toPoint()
    pos = pos + QtCore.QPoint(0, 6)
    pos2 = pos + QtCore.QPoint(-20, -20)
    mouseDrag(plt, pos, pos2, QtCore.Qt.LeftButton)
    assert hline.value() == 0

    # test vertical drag
    pos = plt.plotItem.vb.mapViewToScene(pg.Point(5,0)).toPoint()
    pos2 = pos - QtCore.QPoint(50, 50)
    mouseDrag(plt, pos, pos2, QtCore.Qt.LeftButton)
    px = hline.pixelLength(pg.Point(1, 0), ortho=True)
    assert abs(hline.value() - plt.plotItem.vb.mapSceneToView(pos2).y()) <= px


if __name__ == '__main__':
    test_mouseInteraction()
