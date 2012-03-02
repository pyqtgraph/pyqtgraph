from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.functions as fn

__all__ = ['ArrowItem']
class ArrowItem(QtGui.QGraphicsPolygonItem):
    """
    For displaying scale-invariant arrows.
    For arrows pointing to a location on a curve, see CurveArrow
    
    """
    
    
    def __init__(self, **opts):
        QtGui.QGraphicsPolygonItem.__init__(self)
        defOpts = {
            'style': 'tri',
            'pxMode': True,
            'size': 20,
            'angle': -150,
            'pos': (0,0),
            'width': 8,
            'tipAngle': 25,
            'baseAngle': 90,
            'pen': (200,200,200),
            'brush': (50,50,200),
        }
        defOpts.update(opts)
        
        self.setStyle(**defOpts)
        
        self.setPen(fn.mkPen(defOpts['pen']))
        self.setBrush(fn.mkBrush(defOpts['brush']))
        
        self.rotate(self.opts['angle'])
        self.moveBy(*self.opts['pos'])
    
    def setStyle(self, **opts):
        self.opts = opts
        
        if opts['style'] == 'tri':
            points = [
                QtCore.QPointF(0,0),
                QtCore.QPointF(opts['size'],-opts['width']/2.),
                QtCore.QPointF(opts['size'],opts['width']/2.),
            ]
            poly = QtGui.QPolygonF(points)
            
        else:
            raise Exception("Unrecognized arrow style '%s'" % opts['style'])
        
        self.setPolygon(poly)
        
        if opts['pxMode']:
            self.setFlags(self.flags() | self.ItemIgnoresTransformations)
        else:
            self.setFlags(self.flags() & ~self.ItemIgnoresTransformations)
        
    def paint(self, p, *args):
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        QtGui.QGraphicsPolygonItem.paint(self, p, *args)
