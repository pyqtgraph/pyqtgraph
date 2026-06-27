import re

import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import SVGExporter

app = pg.mkQApp()

def test_plotscene(tmpdir):
    pg.setConfigOption('foreground', (0,0,0))
    w = pg.GraphicsLayoutWidget()
    w.show()        
    p1 = w.addPlot()
    p2 = w.addPlot()
    p1.plot([1,3,2,3,1,6,9,8,4,2,3,5,3], pen={'color':'k'})
    p1.setXRange(0,5)
    p2.plot([1,5,2,3,4,6,1,2,4,2,3,5,3], pen={'color':'k', 'cosmetic':False, 'width': 0.3})
    app.processEvents()
    app.processEvents()
    
    ex = SVGExporter(w.scene())

    tf = tmpdir.join("export.svg")
    ex.export(fileName=tf)
    # clean up after the test is done
    w.close()

def test_simple(tmpdir):    
    view = pg.GraphicsView()
    view.show()

    scene = view.sceneObj

    rect = pg.QtWidgets.QGraphicsRectItem(0, 0, 100, 100)
    scene.addItem(rect)
    rect.setPos(20,20)
    tr = pg.QtGui.QTransform()
    rect.setTransform(tr.translate(50, 50).rotate(30).scale(0.5, 0.5))
    
    rect1 = pg.QtWidgets.QGraphicsRectItem(0, 0, 100, 100)
    rect1.setParentItem(rect)
    rect1.setFlag(rect1.GraphicsItemFlag.ItemIgnoresTransformations)
    rect1.setPos(20, 20)
    rect1.setScale(2)
    
    el1 = pg.QtWidgets.QGraphicsEllipseItem(0, 0, 100, 100)
    el1.setParentItem(rect1)
    grp = pg.ItemGroup()
    grp.setParentItem(rect)
    tr = pg.QtGui.QTransform()
    grp.setTransform(tr.translate(200, 0).rotate(30))
    
    rect2 = pg.QtWidgets.QGraphicsRectItem(0, 0, 100, 25)
    rect2.setFlag(rect2.GraphicsItemFlag.ItemClipsChildrenToShape)
    rect2.setParentItem(grp)
    rect2.setPos(0,25)
    rect2.setRotation(30)
    el = pg.QtWidgets.QGraphicsEllipseItem(0, 0, 100, 50)
    tr = pg.QtGui.QTransform()
    el.setTransform(tr.translate(10, -5).scale(0.5, 2))

    el.setParentItem(rect2)

    grp2 = pg.ItemGroup()
    scene.addItem(grp2)
    grp2.setScale(100)

    rect3 = pg.QtWidgets.QGraphicsRectItem(0,0,2,2)
    rect3.setPen(pg.mkPen(width=1, cosmetic=False))
    grp2.addItem(rect3)

    ex = SVGExporter(scene)
    tf = tmpdir.join("export.svg")
    ex.export(fileName=tf)


def test_large_coordinate_curve_export(tmpdir):
    w = pg.GraphicsLayoutWidget()
    w.show()

    plot = w.addPlot()
    x = np.arange(0, 500, 10, dtype=float) + 10_000_000
    y = np.linspace(0, 1, x.size)
    plot.plot(x=x, y=y, pen="g")

    app.processEvents()
    app.processEvents()

    ex = SVGExporter(w.scene())
    tf = tmpdir.join("export.svg")
    ex.export(fileName=tf)
    w.close()

    text = tf.read_text("utf-8")
    path_coords = []
    for path in re.findall(r'<path[^>]* d="([^"]+)"', text):
        xs = []
        for token in path.strip().split():
            token = token.lstrip("ML")
            if "," not in token:
                continue
            xs.append(float(token.split(",", 1)[0]))
        if xs:
            path_coords.append(xs)

    curve_xs = max(path_coords, key=len)
    assert len(curve_xs) == x.size
    assert np.all(np.diff(curve_xs) > 0)


def test_svg_export_skips_fill_path_list(tmpdir, monkeypatch):
    w = pg.GraphicsLayoutWidget()
    w.show()

    plot = w.addPlot()
    item = plot.plot(
        x=np.arange(10, dtype=float),
        y=np.linspace(1, 2, 10),
        pen="g",
        fillLevel=0,
        brush="g",
    )

    app.processEvents()
    app.processEvents()

    fill_path_list_calls = []
    original = item.curve._getFillPathList

    def record_fill_path_list(widget):
        fill_path_list_calls.append(widget)
        return original(widget)

    monkeypatch.setattr(item.curve, "_getFillPathList", record_fill_path_list)

    ex = SVGExporter(w.scene())
    tf = tmpdir.join("export.svg")
    ex.export(fileName=tf)
    w.close()

    assert fill_path_list_calls == []
