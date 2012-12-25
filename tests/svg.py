"""
SVG export test
"""
import test
import pyqtgraph as pg
app = pg.mkQApp()

class SVGTest(test.TestCase):
    def test_plotscene(self):
        p = pg.plot([1,5,2,3,4,6,1,2,4,2,3,5,3])
        p.setXRange(0,5)
        ex = pg.exporters.SVGExporter.SVGExporter(p.scene())
        ex.export(fileName='test.svg')

    #def test_simple(self):
        #rect = pg.QtGui.QGraphicsRectItem(0, 0, 100, 100)
        ##rect.rotate(30)
        #grp = pg.ItemGroup()
        #grp.setParentItem(rect)
        #grp.translate(200,0)
        #grp.rotate(30)
        #el = pg.QtGui.QGraphicsEllipseItem(10, 0, 100, 50)
        #el.setParentItem(grp)
        #ex = pg.exporters.SVGExporter.SVGExporter(rect)
        #ex.export(fileName='test.svg')
        

if __name__ == '__main__':
    test.unittest.main()