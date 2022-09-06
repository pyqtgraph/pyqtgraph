import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

app = pg.mkQApp("GraphicsScene Example")
win = pg.GraphicsView()
win.show()


class Obj(QtWidgets.QGraphicsObject):
    def __init__(self):
        QtWidgets.QGraphicsObject.__init__(self)
        
    def paint(self, p, *args):
        p.setPen(pg.mkPen(200,200,200))
        p.drawRect(self.boundingRect())
        
    def boundingRect(self):
        return QtCore.QRectF(0, 0, 20, 20)
        
    def mouseClickEvent(self, ev):
        if ev.double():
            print("double click")
        else:
            print("click")
        ev.accept()
        
    #def mouseDragEvent(self, ev):
        #print "drag"
        #ev.accept()
        #self.setPos(self.pos() + ev.pos()-ev.lastPos())
        
        

vb = pg.ViewBox()
win.setCentralItem(vb)

obj = Obj()
vb.addItem(obj)

obj2 = Obj()
win.addItem(obj2)

def clicked():
    print("button click")
btn = QtWidgets.QPushButton("BTN")
btn.clicked.connect(clicked)
prox = QtWidgets.QGraphicsProxyWidget()
prox.setWidget(btn)
prox.setPos(100,0)
vb.addItem(prox)

g = pg.GridItem()
vb.addItem(g)

if __name__ == '__main__':
    pg.exec()
