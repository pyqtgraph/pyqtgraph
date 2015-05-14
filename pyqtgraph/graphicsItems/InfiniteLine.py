# -*- coding: utf-8 -*-

from ..Qt import QtGui, QtCore
from .. import functions as fn
from .UIGraphicsItem import UIGraphicsItem
from .TextItem import TextItem
from .InfiniteLineBase import InfiniteLineBase
from ..Point import Point



__all__ = ['InfiniteLine']
class InfiniteLine(UIGraphicsItem):
    """
    **Bases:** :class:`UIGraphicsItem <pyqtgraph.UIGraphicsItem>`
    
    Displays a line of infinite length with a label showing its location.
    The line can be dragged to indicate a particular position in data coordinates.
    
    ===============================  =============================================================================
    **Signals:**
    sigDragged(self)                
    sigPositionChangeFinish(self)    Emitted when the user has finished dragging the line or when its location is
                                     changed programatically.
    sigPositionChanged(self)         Emitted while the user is dragging the line or when its location is changed
                                     programatically.
    ===============================  =============================================================================
    """

    sigDragged = QtCore.Signal(object)
    sigPositionChangeFinished = QtCore.Signal(object)
    sigPositionChanged = QtCore.Signal(object)
    
    def __init__(self, pos=None, angle=90, pen=None, movable=True, bounds=None, 
                 label=False, textColor=(200,200,200), textBorder=None, textFill=None, 
                 textLocation=0.05, textShift=0.5, textFormat="{:.3f}", unit=None, name=None):
        """Create a new InfiniteLine.
        
        ==============  =====================================================================
        **Arguments:**
        pos             Position of the line. This can be a QPointF or a single value for
                        vertical/horizontal lines.
        angle           Angle of line in degrees. 0 is horizontal, 90 is vertical.
        pen             Pen to use when drawing line. Can be any arguments that are valid
                        for :func:`mkPen <pyqtgraph.mkPen>`. Default pen is transparent
                        yellow.
        movable         If True, the line can be dragged to a new position by the user.
        bounds          Optional [min, max] bounding values. Bounds are only valid if the
                        line is vertical or horizontal.
        label           if True, a label is displayed next to the line to indicate its
                        location in data coordinates
        textColor       color of the label. Can be any argument fn.mkColor can understand.
        textBorder      A Pen to use when drawing the border of the text.
        textFill        A brush to use when filling within the border of the text.
        textLocation    A float [0-1] that defines the location of the text.
        textShift       A float [0-1] that defines when the text shifts from one side to
                        another.
        textFormat      Any new python 3 str.format() format.
        unit            If not None, corresponds to the unit to show next to the label
        name            If not None, corresponds to the name of the object
        ==============  =====================================================================
        """
        
        UIGraphicsItem.__init__(self)
        if angle is None:
            angle = 90 
        self.angle = angle
        self.bounds = bounds
        self.textColor = textColor
        self.location = textLocation
        self.shift = textShift
        self.label = label
        self.format = textFormat
        self.unit = unit
        self.name = name

        self.anchorLeft = (1., 0.5)
        self.anchorRight = (0., 0.5)
        self.anchorUp = (0.5, 1.)
        self.anchorDown = (0.5, 0.)
        
        self.line = InfiniteLineBase(pos=pos, angle=self.angle, pen=pen, 
                        movable=movable, bounds=bounds)
        self.text = TextItem(border=textBorder, fill=textFill)

        if not self.label:
            self.text.hide()

        # important : on s'assurce que les objets sont bien des parents de la classe
        # cela permet de bénéficier de l'ensemble de ses méthodes. 
        self.line.setParentItem(self)
        self.text.setParentItem(self)

        ## Explicitly wrap methods from InfiniteLineBase
        for m in ['value', 'setBounds', 'setMovable', 'setPos']:
            setattr(self, m, getattr(self.line, m))

        # we set the line to be movable or not
        self.setMovable(movable)
            
        self.line.sigPositionChangeFinished.connect(self.lineMoveFinished)
        self.line.sigPositionChanged.connect(self.lineMoved)


    def setValue(self, val):
        """
        Set the location of the line.

        ==============   ==============================================
        **Arguments:**
        val              a float .
        ==============   ==============================================
        """
        if val == self.line.value():
            return
        self.line.setValue(val)
        self.lineMoved()
        self.lineMoveFinished()


    def viewRangeChanged(self):
        self.update()

    def boundingRect(self):
        br = UIGraphicsItem.boundingRect(self)
        return br
        
    def paint(self, p, *args):
        pass

    def lineMoved(self):
        self.prepareGeometryChange()
        self.update()
        self.sigPositionChanged.emit(self)
        self.sigDragged.emit(self)
            
    def lineMoveFinished(self):
        self.update()
        self.sigPositionChangeFinished.emit(self)

    def update(self):
        br = UIGraphicsItem.boundingRect(self)
        xmin, ymin, xmax, ymax = br.getCoords()
        pos = self.line.value()
        if self.angle == 90:  # ligne verticale
            diffX = xmax-xmin
            diffMin = pos-xmin
            limInf = self.shift*diffX
            ypos = ymin+self.location*(ymax-ymin)
            if diffMin < limInf:
                self.text.anchor = Point(self.anchorRight)
            else:
                self.text.anchor = Point(self.anchorLeft)
            fmt = " x = " + self.format  
            if self.unit is not None:
                fmt = fmt + self.unit
            self.text.setText(fmt.format(pos), color=self.textColor)
            self.text.setPos(pos, ypos)
        else:  # ligne horizontale
            diffY = ymax-ymin
            diffMin = pos-ymin
            limInf = self.shift*(ymax-ymin)
            xpos = xmin+self.location*(xmax-xmin)
            if diffMin < limInf:
                self.text.anchor = Point(self.anchorUp)
            else:
                self.text.anchor = Point(self.anchorDown)
            fmt = " y = " + self.format
            if self.unit is not None:
                fmt = fmt + self.unit
            self.text.setText(fmt.format(pos), color = self.textColor) 
            self.text.setPos(xpos, pos)

    def setPen(self, pen):
        """
        Set the pen used to draw the line.

        ==============   ==============================================
        **Arguments:**
        pen              Pen to use when drawing line. Can be any arguments 
                         that are valid for :func:`mkPen <pyqtgraph.mkPen>`.
        ==============   ==============================================
        """
        self.line.setPen(pen)
        self.update()

    def setHoverPen(self, pen):
        """
        Set the pen used to drawingw the line when the mouse is hovering it.

        ==============   ==============================================
        **Arguments:**
        pen              Pen to use when drawing line. Can be any arguments 
                         that are valid for :func:`mkPen <pyqtgraph.mkPen>`.
        ==============   ==============================================
        """
        self.line.setHoverPen(pen)
        self.update()

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
        Display or not the label indicating the location of the line in data
        coordinates.

        ==============   ==============================================
        **Arguments:**
        state            If True, the label is shown. Otherwise, it is hidden.
        ==============   ==============================================
        """
        if state:
            self.text.show()
        else:
            self.text.hide()
        self.update()

    def setLocation(self, loc):
        """
        Set the location of the textItem with respect to a specific axis. If the
        line is vertical, the location is based on the normalized range of the
        yaxis. Otherwise, it is based on the normalized range of the xaxis.

        ==============   ==============================================
        **Arguments:**
        loc              the normalized location of the textItem.
        ==============   ==============================================
        """
        if loc > 1.:
            loc = 1.
        if loc < 0.:
            loc = 0.
        self.location = loc
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

    def setFormat(self, format):
        """
        Set the format of the label used to indicate the location of the line.


        ==============   ==============================================
        **Arguments:**
        format           Any format compatible with the new python 3 
                         str.format() format style.
        ==============   ==============================================
        """
        # todo: needs to check that the given format is good.
        self.format = format
        self.update()

    def setUnit(self, unit):
        """
        Set the unit of the label used to indicate the location of the line.


        ==============   ==============================================
        **Arguments:**
        unit             Any string.
        ==============   ==============================================
        """
        self.unit = unit
        self.update()
    
    def setName(self, name):
        self.name = name
    
    def getName(self):
        return self.name