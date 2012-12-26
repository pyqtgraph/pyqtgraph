"""
SVG export test
"""
import test
import pyqtgraph as pg
app = pg.mkQApp()

class SVGTest(test.TestCase):
    def test_plotscene(self):
        pg.setConfigOption('foreground', (0,0,0))
        w = pg.GraphicsWindow()
        w.show()        
        p1 = w.addPlot()
        p2 = w.addPlot()
        p1.plot([1,3,2,3,1,6,9,8,4,2,3,5,3], pen={'color':'k'})
        p1.setXRange(0,5)
        p2.plot([1,5,2,3,4,6,1,2,4,2,3,5,3], pen={'color':'k', 'cosmetic':False, 'width': 0.3})
        app.processEvents()
        app.processEvents()
        
        ex = pg.exporters.SVGExporter.SVGExporter(w.scene())
        ex.export(fileName='test.svg')

    #def test_simple(self):
        #rect = pg.QtGui.QGraphicsRectItem(0, 0, 100, 100)
        #rect.translate(50, 50)
        #rect.rotate(30)
        #grp = pg.ItemGroup()
        #grp.setParentItem(rect)
        #grp.translate(200,0)
        ##grp.rotate(30)
        
        #rect2 = pg.QtGui.QGraphicsRectItem(0, 0, 100, 25)
        #rect2.setFlag(rect2.ItemClipsChildrenToShape)
        #rect2.setParentItem(grp)
        #rect2.setPos(0,25)
        #rect2.rotate(30)
        #el = pg.QtGui.QGraphicsEllipseItem(0, 0, 100, 50)
        #el.translate(10,-5)
        #el.scale(0.5,2)
        #el.setParentItem(rect2)
        
        #ex = pg.exporters.SVGExporter.SVGExporter(rect)
        #ex.export(fileName='test.svg')
        

if __name__ == '__main__':
    test.unittest.main()