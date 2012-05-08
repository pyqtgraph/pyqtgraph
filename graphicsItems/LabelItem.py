from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.functions as fn
from GraphicsWidget import GraphicsWidget


__all__ = ['LabelItem']

class LabelItem(GraphicsWidget):
    """
    GraphicsWidget displaying text.
    Used mainly as axis labels, titles, etc.
    
    Note: To display text inside a scaled view (ViewBox, PlotWidget, etc) use TextItem
    """
    
    
    def __init__(self, text=' ', parent=None, angle=0, **args):
        GraphicsWidget.__init__(self, parent)
        self.item = QtGui.QGraphicsTextItem(self)
        self.opts = {
            'color': 'CCC',
            'justify': 'center'
        }
        self.opts.update(args)
        self.sizeHint = {}
        self.setText(text)
        self.setAngle(angle)
        
            
    def setAttr(self, attr, value):
        """Set default text properties. See setText() for accepted parameters."""
        self.opts[attr] = value
        
    def setText(self, text, **args):
        """Set the text and text properties in the label. Accepts optional arguments for auto-generating
        a CSS style string:

        ==================== ==============================
        **Style Arguments:**
        color                (str) example: 'CCFF00'
        size                 (str) example: '8pt'
        bold                 (bool)
        italic               (bool)
        ==================== ==============================
        """
        self.text = text
        opts = self.opts.copy()
        for k in args:
            opts[k] = args[k]
        
        optlist = []
        if 'color' in opts:
            if isinstance(opts['color'], QtGui.QColor):
                opts['color'] = fn.colorStr(opts['color'])[:6]
            optlist.append('color: #' + opts['color'])
        if 'size' in opts:
            optlist.append('font-size: ' + opts['size'])
        if 'bold' in opts and opts['bold'] in [True, False]:
            optlist.append('font-weight: ' + {True:'bold', False:'normal'}[opts['bold']])
        if 'italic' in opts and opts['italic'] in [True, False]:
            optlist.append('font-style: ' + {True:'italic', False:'normal'}[opts['italic']])
        full = "<span style='%s'>%s</span>" % ('; '.join(optlist), text)
        #print full
        self.item.setHtml(full)
        self.updateMin()
        self.resizeEvent(None)
        self.update()
        
    def resizeEvent(self, ev):
        #c1 = self.boundingRect().center()
        #c2 = self.item.mapToParent(self.item.boundingRect().center()) # + self.item.pos()
        #dif = c1 - c2
        #self.item.moveBy(dif.x(), dif.y())
        #print c1, c2, dif, self.item.pos()
        if self.opts['justify'] == 'left':
            self.item.setPos(0,0)
        elif self.opts['justify'] == 'center':
            bounds = self.item.mapRectToParent(self.item.boundingRect())
            self.item.setPos(self.width()/2. - bounds.width()/2., 0)
        elif self.opts['justify'] == 'right':
            bounds = self.item.mapRectToParent(self.item.boundingRect())
            self.item.setPos(self.width() - bounds.width(), 0)
        #if self.width() > 0:
            #self.item.setTextWidth(self.width())
        
    def setAngle(self, angle):
        self.angle = angle
        self.item.resetTransform()
        self.item.rotate(angle)
        self.updateMin()
        
    def updateMin(self):
        bounds = self.item.mapRectToParent(self.item.boundingRect())
        self.setMinimumWidth(bounds.width())
        self.setMinimumHeight(bounds.height())
        
        self.sizeHint = {
            QtCore.Qt.MinimumSize: (bounds.width(), bounds.height()),
            QtCore.Qt.PreferredSize: (bounds.width(), bounds.height()),
            QtCore.Qt.MaximumSize: (-1, -1),  #bounds.width()*2, bounds.height()*2),
            QtCore.Qt.MinimumDescent: (0, 0)  ##?? what is this?
        }
            
        self.update()
        
    def sizeHint(self, hint, constraint):
        if hint not in self.sizeHint:
            return QtCore.QSizeF(0, 0)
        return QtCore.QSizeF(*self.sizeHint[hint])
        
    #def paint(self, p, *args):
        #p.setPen(fn.mkPen('r'))
        #p.drawRect(self.rect())
        #p.drawRect(self.item.boundingRect())
        
