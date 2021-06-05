import pyqtgraph as pg


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
    
    ex = pg.exporters.SVGExporter(w.scene())

    tf = tmpdir.join("expot.svg")
    ex.export(fileName=tf)
    # clean up after the test is done
    w.close()

def test_simple(tmpdir):    
    view = pg.GraphicsView()
    view.show()

    scene = view.sceneObj

    rect = pg.QtGui.QGraphicsRectItem(0, 0, 100, 100)
    scene.addItem(rect)
    rect.setPos(20,20)
    tr = pg.QtGui.QTransform()
    rect.setTransform(tr.translate(50, 50).rotate(30).scale(0.5, 0.5))
    
    rect1 = pg.QtGui.QGraphicsRectItem(0, 0, 100, 100)
    rect1.setParentItem(rect)
    rect1.setFlag(rect1.ItemIgnoresTransformations)
    rect1.setPos(20, 20)
    rect1.setScale(2)
    
    el1 = pg.QtGui.QGraphicsEllipseItem(0, 0, 100, 100)
    el1.setParentItem(rect1)
    grp = pg.ItemGroup()
    grp.setParentItem(rect)
    tr = pg.QtGui.QTransform()
    grp.setTransform(tr.translate(200, 0).rotate(30))
    
    rect2 = pg.QtGui.QGraphicsRectItem(0, 0, 100, 25)
    rect2.setFlag(rect2.ItemClipsChildrenToShape)
    rect2.setParentItem(grp)
    rect2.setPos(0,25)
    rect2.setRotation(30)
    el = pg.QtGui.QGraphicsEllipseItem(0, 0, 100, 50)
    tr = pg.QtGui.QTransform()
    el.setTransform(tr.translate(10, -5).scale(0.5, 2))

    el.setParentItem(rect2)

    grp2 = pg.ItemGroup()
    scene.addItem(grp2)
    grp2.setScale(100)

    rect3 = pg.QtGui.QGraphicsRectItem(0,0,2,2)
    rect3.setPen(pg.mkPen(width=1, cosmetic=False))
    grp2.addItem(rect3)

    ex = pg.exporters.SVGExporter(scene)
    tf = tmpdir.join("expot.svg")
    ex.export(fileName=tf)
