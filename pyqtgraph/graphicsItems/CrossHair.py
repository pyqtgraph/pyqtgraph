# -*- coding: utf-8 -*-

from ..Qt import QtGui, QtCore
from .UIGraphicsItem import UIGraphicsItem
from .. import functions as fn
from .TextItem import TextItem
from .InfiniteLineBase import InfiniteLineBase
from ..Point import Point


__all__ = ['CrossHair']

class CrossHair(UIGraphicsItem):
    """
    **Bases:** :class:`UIGraphicsItem <pyqtgraph.UIGraphicsItem>`
    
    Used for marking a specific region in plots using a vertical and a
    horizontal line. 
    The item can be dragged to point a specific location in the plot.
    
    ===============================  =============================================================================
    **Signals:**
    sigPositionChangeFinished(self)  Emitted when the user has finished dragging the item and when the item 
                                     is changed programatically.
    sigPositionChanged(self)         Emitted while the user is dragging the item and when the item is 
                                     changed programatically.
    ===============================  =============================================================================
    """
    
    sigPositionChangeFinished = QtCore.pyqtSignal()
    sigPositionChanged = QtCore.pyqtSignal()
    
    def __init__(self, values=(None,None), pen=None, movable=True, bounds=(None, None),
                 label=True, textColor=(200,200,200), textBorder=None, 
                 textFill=None, textShift=0.5, textFormat=("{:.3f}","{:.3f}"),
                 units=(None, None), name=None):
        """Create a new CrossHair item.
        
        ==============  =====================================================================
        **Arguments:**
        values             A list of the positions of the lines in the region. These are not
                        limits; limits can be set by specifying bounds.
        pen             Pen to use when drawing lines. Can be any arguments that are valid
                        for :func:`mkPen <pyqtgraph.mkPen>`. Default pen is transparent
                        yellow.
        movable         If True, the region and individual lines are movable by the user; if
                        False, they are static.
        bounds          Optional [(xmin, xmax), (ymin, ymax)] bounding values for the item
        label           if True, a label is displayed next to the line to indicate its
                        location in data coordinates
        textColor       color of the label. Can be any argument fn.mkColor can understand.
        textBorder      A Pen to use when drawing the border of the text.
        textFill        A brush to use when filling within the border of the text.
        textLocation    A float [0-1] that defines the location of the text.
        textShift       A float [0-1] that defines when the text shifts from one side to
                        another.
        textFormat      Any new python 3 list of str.format() format.
        units           If not None, corresponds to the unit to show next to the label 
        name            If not None, corresponds to the name of the object
        ==============  =====================================================================
        """
        
        UIGraphicsItem.__init__(self)

        self.bounds = bounds
        self.textColor = textColor
        self.shift = textShift
        self.format = textFormat
        self.label = label
        self.units = units
        self.name = name

        self.blockLineSignal = False
        self.moving = False
        self.mouseHovering = False

        self.anchorTopLeft = (1.,1.)
        self.anchorTopRight = (0.,1.)
        self.anchorBottomLeft = (1.,0.)
        self.anchorBottomRight = (0.,0.)
        
        self.hline = InfiniteLineBase(pos=values[0], angle=0, pen=pen, 
                        movable=movable, bounds=bounds[0])
        self.vline = InfiniteLineBase(pos=values[1], angle=90, pen=pen, 
                        movable=movable, bounds=bounds[1])
        self.text = TextItem(border=textBorder, fill=textFill)

        if not self.label:
            self.text.hide()

        self.hline.setParentItem(self)
        self.vline.setParentItem(self)
        self.text.setParentItem(self)

        self.setMovable(movable)
            
        self.hline.sigPositionChangeFinished.connect(self.linesMoveFinished)
        self.vline.sigPositionChangeFinished.connect(self.linesMoveFinished)
        self.hline.sigPositionChanged.connect(self.linesMoved)
        self.vline.sigPositionChanged.connect(self.linesMoved)

        
    def value(self):
        """ Returns the current position of the crosshair """
        return self.hline.value(), self.vline.value()

    def setValue(self, val):
        """
        Set the location of the crosshair.

        ==============   ==============================================
        **Arguments:**
        val              A list or tuple of the x location and y location
                         of the crosshair.
        ==============   ==============================================
        """
        if val[0] == self.hline.value() and val[1] == self.vline.value():
            return
        self.blockLineSignal = True
        self.vline.setValue(val[0])
        self.blockLineSignal = False
        self.hline.setValue(val[1])
        self.linesMoved()
        self.linesMoveFinished()

    def setBounds(self, bounds):
        """
        Optional [min, max] bounding values for the region 

        ==============   ==============================================
        **Arguments:**
        bounds           A list or tuple of the bounds of each lines starting
                        from the vertical line
        ==============   ==============================================
        """
        self.vline.setBounds(bounds[0])
        self.hline.setBounds(bounds[1])
        
    def setMovable(self, state):
        """
        Set lines to be movable by the user, or not.
        """
        self.vline.setMovable(state)
        self.hline.setMovable(state)

    def viewRangeChanged(self):
        self.update()

    def isMoving(self):
        self.moving = True

    def stopMoving(self):
        self.moving = False

    def boundingRect(self):
        br = UIGraphicsItem.boundingRect(self)
        return br
        
    def paint(self, p, *args):
        # ugly thing. Prettier way to do it?
        self.getViewBox().scene().sigMouseMoved.connect(self.updatePosition)

    def linesMoved(self):
        self.isMoving()
        if self.blockLineSignal:
            return
        self.prepareGeometryChange()
        self.update()
        self.sigPositionChanged.emit()
            
    def linesMoveFinished(self):
        self.stopMoving()
        self.update()
        self.sigPositionChangeFinished.emit()

    def updatePosition(self, pos):
        if not self.moving:
            return
        if self.getViewBox().sceneBoundingRect().contains(pos):
            mousePoint = self.getViewBox().mapSceneToView(pos)
            self.setValue([mousePoint.x(), mousePoint.y()])

    def update(self):
        br = UIGraphicsItem.boundingRect(self)
        xmin, ymin, xmax, ymax = br.getCoords()
        posX = self.vline.value()
        posY = self.hline.value()
        diffXMin = posX-xmin
        diffYMin = posY-ymin
        limX = self.shift*(xmax-xmin)
        limY = self.shift*(ymax-ymin)
        if diffXMin < limX:
            if diffYMin < limY:
                self.text.anchor = Point(self.anchorTopRight)
            else:
                self.text.anchor = Point(self.anchorBottomRight)
        else:
            if diffYMin < limY:
                self.text.anchor = Point(self.anchorTopLeft)
            else:
                self.text.anchor = Point(self.anchorBottomLeft)
        fmt = " x = "+self.format[0]
        if self.units[0] is not None:
            fmt = fmt + self.units[0]
        fmt = fmt + "\n y = "+self.format[1]
        if self.units[1] is not None:
            fmt = fmt + self.units[1]
        self.text.setText(fmt.format(posX, posY), color=self.textColor)
        self.text.setPos(posX, posY)

    def getPen(self):
        """ Returns the current pen in use (the same for each line). """
        return self.hline.pen

    def getHoverPen(self):
        """ Returns the current hover pen in use (the same for each line). """
        return self.hline.hoverPen

    def setPen(self, *args, **kwargs):
        """
        Set the pen used to draw the line. Allowable arguments are any that are
        valid for :func:`mkPen <pyqtgraph.mkPen>`.
        """
        self.hline.setPen(pen)
        self.vline.setPen(pen)

    def setHoverPen(self, *args, **kwargs):
        """
        Set the pen used to drawingw the line when the mouse is hovering it.
        Allowable arguments are any that are valid for 
        :func:`mkPen <pyqtgraph.mkPen>`.
        """
        self.hline.setHoverPen(pen)
        self.vline.setHoverPen(pen)

    def setTextColor(self, color):
        """
        Set the color of the label.

        ==============   ==============================================
        **Arguments:**
        color            Can be any arguments that are valid for 
                         :func:`mkColor <pyqtgraph.mkColor>`.
        ==============   ==============================================
        """
        self.textColor = color
        self.update()

    def showLabel(self, state):
        """
        Display or not the label indicating the location of the lines in data
        coordinates.

        ==============   ==============================================
        **Arguments:**
        state            If True, the labels are shown. Otherwise, they
                         are hidden.
        ==============   ==============================================
        """
        if state:
            self.text.show()
        else:
            self.text.hide()
        self.update()

    def setShift(self, shift):
        """
        Set the value with respect to the normalized range of the corresponding
        axis where the location of the textItem shifts from one side to another.

        ==============   ==============================================
        **Arguments:**
        shift              the normalized shift value of the textItem.
        ==============   ==============================================
        """
        if shift > 1.:
            shift = 1.
        if shift < 0.:
            shift = 0.
        self.shift = shift
        self.update()

    def setFormat(self, x="{:.3f}", y="{:.3f}"):
        """
        Set the format of the label used to indicate the location of the line.


        ==============   ==============================================
        **Arguments:**
        x                Any format compatible with the new python 3 
                         str.format() format style.
        y                Any format compatible with the new python 3 
                         str.format() format style.
        ==============   ==============================================
        """
        self.format = (x, y)
        self.update()

    def setUnits(self, x=None, y=None):
        """
        Set the units of the labels used to indicate the location of the item.


        ==============   ==============================================
        **Arguments:**
        units            A list containing the x unit and the y unit.
        ==============   ==============================================
        """
        self.units = (x, y)

    def setName(self, name):
        self.name = name
    
    def getName(self):
        return self.name