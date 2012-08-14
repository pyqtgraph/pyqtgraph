from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from .UIGraphicsItem import *
import pyqtgraph.functions as fn

class TextItem(UIGraphicsItem):
    """
    GraphicsItem displaying unscaled text (the text will always appear normal even inside a scaled ViewBox). 
    """
    def __init__(self, text='', color=(200,200,200), html=None, anchor=(0,0), border=None, fill=None):
        """
        Arguments:
        *text*   The text to display 
        *color*  The color of the text (any format accepted by pg.mkColor)
        *html*   If specified, this overrides both *text* and *color*
        *anchor* A QPointF or (x,y) sequence indicating what region of the text box will 
                 be anchored to the item's position. A value of (0,0) sets the upper-left corner
                 of the text box to be at the position specified by setPos(), while a value of (1,1)
                 sets the lower-right corner.
        *border* A pen to use when drawing the border
        *fill*   A brush to use when filling within the border
        """
        UIGraphicsItem.__init__(self)
        self.textItem = QtGui.QGraphicsTextItem()
        self.lastTransform = None
        self._bounds = QtCore.QRectF()
        if html is None:
            self.setText(text, color)
        else:
            self.setHtml(html)
        self.anchor = pg.Point(anchor)
        self.fill = pg.mkBrush(fill)
        self.border = pg.mkPen(border)
        #self.setFlag(self.ItemIgnoresTransformations)  ## This is required to keep the text unscaled inside the viewport

    def setText(self, text, color=(200,200,200)):
        color = pg.mkColor(color)
        self.textItem.setDefaultTextColor(color)
        self.textItem.setPlainText(text)
        #html = '<span style="color: #%s; text-align: center;">%s</span>' % (color, text)
        #self.setHtml(html)
        
    def updateAnchor(self):
        pass
        #self.resetTransform()
        #self.translate(0, 20)
        
    def setPlainText(self, *args):
        self.textItem.setPlainText(*args)
        self.updateText()
        
    def setHtml(self, *args):
        self.textItem.setHtml(*args)
        self.updateText()
        
    def setTextWidth(self, *args):
        self.textItem.setTextWidth(*args)
        self.updateText()
        
    def setFont(self, *args):
        self.textItem.setFont(*args)
        self.updateText()
        
    def updateText(self):
        self.viewRangeChanged()

    #def getImage(self):
        #if self.img is None:
            #br = self.textItem.boundingRect()
            #img = QtGui.QImage(int(br.width()), int(br.height()), QtGui.QImage.Format_ARGB32)
            #p = QtGui.QPainter(img)
            #self.textItem.paint(p, QtGui.QStyleOptionGraphicsItem(), None)
            #p.end()
            #self.img = img
        #return self.img
        
    def textBoundingRect(self):
        ## return the bounds of the text box in device coordinates
        pos = self.mapToDevice(QtCore.QPointF(0,0))
        if pos is None:
            return None
        tbr = self.textItem.boundingRect()
        return QtCore.QRectF(pos.x() - tbr.width()*self.anchor.x(), pos.y() - tbr.height()*self.anchor.y(), tbr.width(), tbr.height())


    def viewRangeChanged(self):
        br = self.textBoundingRect()
        if br is None:
            return
        self.prepareGeometryChange()
        self._bounds = fn.invertQTransform(self.deviceTransform()).mapRect(br)
        #print self._bounds

    def boundingRect(self):
        return self._bounds
        
    def paint(self, p, *args):
        tr = p.transform()
        if self.lastTransform is not None:
            if tr != self.lastTransform:
                self.viewRangeChanged()
        self.lastTransform = tr
        
        
        tbr = self.textBoundingRect()
        
        #p.setPen(pg.mkPen('r'))
        #p.drawRect(self.boundingRect())
        
        p.setPen(self.border)
        p.setBrush(self.fill)
        
        
        #p.fillRect(tbr)
        p.resetTransform()
        p.drawRect(tbr)
        
        
        p.translate(tbr.left(), tbr.top())
        self.textItem.paint(p, QtGui.QStyleOptionGraphicsItem(), None)
        