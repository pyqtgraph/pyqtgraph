from .. import functions as fn
from ..Qt import QtCore, QtGui
from .UIGraphicsItem import UIGraphicsItem

__all__ = ['GradientLegend']

class GradientLegend(UIGraphicsItem):
    """
    Draws a color gradient rectangle along with text labels denoting the value at specific
    points along the gradient.
    """
    
    def __init__(self, size, offset):
        self.size = size
        self.offset = offset
        UIGraphicsItem.__init__(self)
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.brush = QtGui.QBrush(QtGui.QColor(255,255,255,100)) # background color
        self.pen = QtGui.QPen(QtGui.QColor(0,0,0))
        self.textPen = QtGui.QPen(QtGui.QColor(0,0,0))
        self.labels = {'max': 1, 'min': 0}
        self.gradient = QtGui.QLinearGradient()
        self.gradient.setColorAt(0, QtGui.QColor(0,0,0))
        self.gradient.setColorAt(1, QtGui.QColor(255,0,0))
        self.setZValue(100) # draw on top of ordinary plots
        
    def setGradient(self, g):
        self.gradient = g
        self.update()
        
    def setColorMap(self, colormap):
        """
        Set displayed gradient from a :class:`~pyqtgraph.ColorMap` object.
        """
        self.gradient = colormap.getGradient()
        
    def setIntColorScale(self, minVal, maxVal, *args, **kargs):
        colors = [fn.intColor(i, maxVal-minVal, *args, **kargs) for i in range(minVal, maxVal)]
        g = QtGui.QLinearGradient()
        for i in range(len(colors)):
            x = float(i)/len(colors)
            g.setColorAt(x, colors[i])
        self.setGradient(g)
        if 'labels' not in kargs:
            self.setLabels({str(minVal): 0, str(maxVal): 1})
        else:
            self.setLabels({kargs['labels'][0]:0, kargs['labels'][1]:1})
        
    def setLabels(self, l):
        """Defines labels to appear next to the color scale. Accepts a dict of {text: value} pairs"""
        self.labels = l
        self.update()
        
    def paint(self, p, opt, widget):
        UIGraphicsItem.paint(self, p, opt, widget)

        view = self.getViewBox()
        if view is None:
            return
        p.save() # save painter state before we change transformation
        trans = view.sceneTransform()
        p.setTransform( trans ) # draw in ViewBox pixel coordinates
        rect = view.rect()
        
        ## determine max width of all labels
        labelWidth = 0
        labelHeight = 0
        for k in self.labels:
            b = p.boundingRect(QtCore.QRectF(0, 0, 0, 0), QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, str(k))
            labelWidth = max(labelWidth, b.width())
            labelHeight = max(labelHeight, b.height())
            
        textPadding = 2  # in px
        
        xR = rect.right()
        xL = rect.left()
        yT = rect.top()
        yB = rect.bottom()
        
        # coordinates describe edges of text and bar, additional margins will be added for background
        if self.offset[0] < 0:
            x3 = xR + self.offset[0] # right edge from right edge of view, offset is negative!
            x2 = x3 - labelWidth - 2*textPadding # right side of color bar
            x1 = x2 - self.size[0]               # left side of color bar
        else:
            x1 = xL + self.offset[0] # left edge from left edge of view
            x2 = x1 + self.size[0]
            x3 = x2 + labelWidth + 2*textPadding # leave room for 2x textpadding between bar and text
        if self.offset[1] < 0:
            y2 = yB + self.offset[1] # bottom edge from bottom of view, offset is negative!
            y1 = y2 - self.size[1]
        else:
            y1 = yT + self.offset[1] # top edge from top of view
            y2 = y1 + self.size[1]
        self.b = [x1,x2,x3,y1,y2,labelWidth]

        ## Draw background
        p.setPen(self.pen)
        p.setBrush(self.brush) # background color        
        rect = QtCore.QRectF(
            QtCore.QPointF(x1 - textPadding, y1-labelHeight/2 - textPadding), # extra left/top padding 
            QtCore.QPointF(x3 + textPadding, y2+labelHeight/2 + textPadding)  # extra bottom/right padding
        )
        p.drawRect(rect)

        ## Draw color bar
        self.gradient.setStart(0, y2) 
        self.gradient.setFinalStop(0, y1)
        p.setBrush(self.gradient)
        rect = QtCore.QRectF(
            QtCore.QPointF(x1, y1),
            QtCore.QPointF(x2, y2)
        )
        p.drawRect(rect)

        ## draw labels
        p.setPen(self.textPen)
        tx = x2 + 2 * textPadding # margin between bar and text
        lh = labelHeight
        lw = labelWidth
        for k in self.labels:
            y = y2 - self.labels[k] * (y2-y1)
            p.drawText(QtCore.QRectF(tx, y - lh/2, lw, lh), QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, str(k))

        p.restore() # restore QPainter transform to original state
