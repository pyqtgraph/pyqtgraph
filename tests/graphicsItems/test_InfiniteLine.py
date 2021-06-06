# -*- coding: utf-8 -*-
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtTest
from tests.ui_testing import mouseDrag, mouseMove
pg.mkQApp()


def test_InfiniteLine():
    # disable delay of mouse move events because events is called immediately in test
    pg.setConfigOption('mouseRateLimit', -1)

    # Test basic InfiniteLine API
    plt = pg.plot()
    plt.setXRange(-10, 10)
    plt.setYRange(-10, 10)
    plt.resize(600, 600)

    # seemingly arbitrary requirements; might need longer wait time for some platforms..
    QtTest.QTest.qWaitForWindowExposed(plt)
    QtTest.QTest.qWait(100)

    vline = plt.addLine(x=1)
    assert vline.angle == 90
    br = vline.mapToView(QtGui.QPolygonF(vline.boundingRect()))
    assert br.containsPoint(pg.Point(1, 5), QtCore.Qt.OddEvenFill)
    assert not br.containsPoint(pg.Point(5, 0), QtCore.Qt.OddEvenFill)
    hline = plt.addLine(y=0)
    assert hline.angle == 0
    assert hline.boundingRect().contains(pg.Point(5, 0))
    assert not hline.boundingRect().contains(pg.Point(0, 5))

    vline.setValue(2)
    assert vline.value() == 2
    vline.setPos(pg.Point(4, -5))
    assert vline.value() == 4

    oline = pg.InfiniteLine(angle=30)
    plt.addItem(oline)
    oline.setPos(pg.Point(1, -1))
    assert oline.angle == 30
    assert oline.pos() == pg.Point(1, -1)
    assert oline.value() == [1, -1]

    # test bounding rect for oblique line
    br = oline.mapToScene(oline.boundingRect())
    pos = oline.mapToScene(pg.Point(2, 0))
    assert br.containsPoint(pos, QtCore.Qt.OddEvenFill)
    px = pg.Point(-0.5, -1.0 / 3**0.5)
    assert br.containsPoint(pos + 5 * px, QtCore.Qt.OddEvenFill)
    assert not br.containsPoint(pos + 7 * px, QtCore.Qt.OddEvenFill)
    plt.close()

def test_mouseInteraction():
    # disable delay of mouse move events because events is called immediately in test
    pg.setConfigOption('mouseRateLimit', -1)

    plt = pg.plot()
    plt.scene().minDragTime = 0  # let us simulate mouse drags very quickly.
    vline = plt.addLine(x=0, movable=True)
    hline = plt.addLine(y=0, movable=True)
    hline2 = plt.addLine(y=-1, movable=False)
    plt.setXRange(-10, 10)
    plt.setYRange(-10, 10)

    # test horizontal drag
    pos = plt.plotItem.vb.mapViewToScene(pg.Point(0,5))
    pos2 = pos - QtCore.QPointF(200, 200)
    mouseMove(plt, pos)
    assert vline.mouseHovering is True and hline.mouseHovering is False
    mouseDrag(plt, pos, pos2, QtCore.Qt.LeftButton)
    px = vline.pixelLength(pg.Point(1, 0), ortho=True)
    assert abs(vline.value() - plt.plotItem.vb.mapSceneToView(pos2).x()) <= px

    # test missed drag
    pos = plt.plotItem.vb.mapViewToScene(pg.Point(5,0))
    pos = pos + QtCore.QPointF(0, 6)
    pos2 = pos + QtCore.QPointF(-20, -20)
    mouseMove(plt, pos)
    assert vline.mouseHovering is False and hline.mouseHovering is False
    mouseDrag(plt, pos, pos2, QtCore.Qt.LeftButton)
    assert hline.value() == 0

    # test vertical drag
    pos = plt.plotItem.vb.mapViewToScene(pg.Point(5,0))
    pos2 = pos - QtCore.QPointF(50, 50)
    mouseMove(plt, pos)
    assert vline.mouseHovering is False and hline.mouseHovering is True
    mouseDrag(plt, pos, pos2, QtCore.Qt.LeftButton)
    px = hline.pixelLength(pg.Point(1, 0), ortho=True)
    assert abs(hline.value() - plt.plotItem.vb.mapSceneToView(pos2).y()) <= px

    # test non-interactive line
    pos = plt.plotItem.vb.mapViewToScene(pg.Point(5,-1))
    pos2 = pos - QtCore.QPointF(50, 50)
    mouseMove(plt, pos)
    assert hline2.mouseHovering == False
    mouseDrag(plt, pos, pos2, QtCore.Qt.LeftButton)
    assert hline2.value() == -1
    plt.close()
