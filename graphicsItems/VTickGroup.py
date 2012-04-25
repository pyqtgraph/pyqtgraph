if __name__ == '__main__':
    import os, sys
    path = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, os.path.join(path, '..', '..'))

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.functions as fn
import weakref
from UIGraphicsItem import UIGraphicsItem

__all__ = ['VTickGroup']
class VTickGroup(UIGraphicsItem):
    """
    **Bases:** :class:`UIGraphicsItem <pyqtgraph.UIGraphicsItem>`
    
    Draws a set of tick marks which always occupy the same vertical range of the view,
    but have x coordinates relative to the data within the view.
    
    """
    def __init__(self, xvals=None, yrange=None, pen=None):
        """
        ============= ===================================================================
        **Arguments**
        xvals         A list of x values (in data coordinates) at which to draw ticks.
        yrange        A list of [low, high] limits for the tick. 0 is the bottom of 
                      the view, 1 is the top. [0.8, 1] would draw ticks in the top 
                      fifth of the view.
        pen           The pen to use for drawing ticks. Default is grey. Can be specified
                      as any argument valid for :func:`mkPen<pyqtgraph.mkPen>`
        ============= ===================================================================
        """
        if yrange is None:
            yrange = [0, 1]
        if xvals is None:
            xvals = []
            
        #bounds = QtCore.QRectF(0, yrange[0], 1, yrange[1]-yrange[0])
        UIGraphicsItem.__init__(self)#, bounds=bounds)
            
        if pen is None:
            pen = (200, 200, 200)
            
        self.path = QtGui.QGraphicsPathItem()
        
        self.ticks = []
        self.xvals = []
        #if view is None:
            #self.view = None
        #else:
            #self.view = weakref.ref(view)
        self.yrange = [0,1]
        self.setPen(pen)
        self.setYRange(yrange)
        self.setXVals(xvals)
        #self.valid = False
        
    def setPen(self, *args, **kwargs):
        """Set the pen to use for drawing ticks. Can be specified as any arguments valid
        for :func:`mkPen<pyqtgraph.mkPen>`"""        
        self.pen = fn.mkPen(*args, **kwargs)

    def setXVals(self, vals):
        """Set the x values for the ticks. 
        
        ============= =====================================================================
        **Arguments** 
        vals          A list of x values (in data/plot coordinates) at which to draw ticks.
        ============= =====================================================================
        """
        self.xvals = vals
        self.rebuildTicks()
        #self.valid = False
        
    def setYRange(self, vals):
        """Set the y range [low, high] that the ticks are drawn on. 0 is the bottom of 
        the view, 1 is the top."""
        self.yrange = vals
        #self.relative = relative
        #if self.view is not None:
            #if relative:
                #self.view().sigRangeChanged.connect(self.rescale)
            #else:
                #try:
                    #self.view().sigRangeChanged.disconnect(self.rescale)
                #except:
                    #pass
        self.rebuildTicks()
        #self.valid = False
        
    def dataBounds(self, *args, **kargs):
        return None  ## item should never affect view autoscaling
            
    #def viewRangeChanged(self):
        ### called when the view is scaled
        
        #UIGraphicsItem.viewRangeChanged(self)
        
        #self.resetTransform()
        ##vb = self.view().viewRect()
        ##p1 = vb.bottom() - vb.height() * self.yrange[0]
        ##p2 = vb.bottom() - vb.height() * self.yrange[1]
        
        ##br = self.boundingRect()
        ##yr = [p1, p2]
        
        
        
        ##self.rebuildTicks()
        
        ##br = self.boundingRect()
        ##print br
        ##self.translate(0.0, br.y())
        ##self.scale(1.0, br.height())
        ##self.boundingRect()
        #self.update()
            
    #def boundingRect(self):
        #print "--request bounds:"
        #b = self.path.boundingRect()
        #b2 = UIGraphicsItem.boundingRect(self)
        #b2.setY(b.y())
        #b2.setWidth(b.width())
        #print "  ", b
        #print "  ", b2
        #print "  ", self.mapRectToScene(b)
        #return b2
            
    def yRange(self):
        #if self.relative:
            #height = self.view.size().height()
            #p1 = self.mapFromScene(self.view.mapToScene(QtCore.QPoint(0, height * (1.0-self.yrange[0]))))
            #p2 = self.mapFromScene(self.view.mapToScene(QtCore.QPoint(0, height * (1.0-self.yrange[1]))))
            #return [p1.y(), p2.y()]
        #else:
            #return self.yrange
            
        return self.yrange
            
    def rebuildTicks(self):
        self.path = QtGui.QPainterPath()
        yrange = self.yRange()
        #print "rebuild ticks:", yrange
        for x in self.xvals:
            #path.moveTo(x, yrange[0])
            #path.lineTo(x, yrange[1])
            self.path.moveTo(x, 0.)
            self.path.lineTo(x, 1.)
        #self.setPath(self.path)
        #self.valid = True
        #self.rescale()
        #print "  done..", self.boundingRect()
        
    def paint(self, p, *args):
        UIGraphicsItem.paint(self, p, *args)
        
        br = self.boundingRect()
        h = br.height()
        br.setY(br.y() + self.yrange[0] * h)
        br.setHeight(h - (1.0-self.yrange[1]) * h)
        p.translate(0, br.y())
        p.scale(1.0, br.height())
        p.setPen(self.pen)
        p.drawPath(self.path)
        #QtGui.QGraphicsPathItem.paint(self, *args)


if __name__ == '__main__':
    app = QtGui.QApplication([])
    import pyqtgraph as pg
    vt = VTickGroup([1,3,4,7,9], [0.8, 1.0])
    p = pg.plot()
    p.addItem(vt)
    
    if sys.flags.interactive == 0:
        app.exec_()
    
    
    
    
    