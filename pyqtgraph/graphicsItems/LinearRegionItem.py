# -*- coding: utf-8 -*-


from ..Qt import QtGui, QtCore
from .UIGraphicsItem import UIGraphicsItem
from .. import functions as fn
from .TextItem import TextItem
from .LinearRegionItemBase import LinearRegionItemBase
from ..Point import Point
from .. import functions as fn
from .InfiniteLineBase import InfiniteLineBase

"""
TODO: some methods are the same as in InfiniteLine.py (setLocation, setShift,
      setFormat). It could be interesting to find a way to circumvent this.
"""

__all__ = ['LinearRegionItem']

class LinearRegionItem(UIGraphicsItem):
    """
    **Bases:** :class:`UIGraphicsItem <pyqtgraph.UIGraphicsItem>`
    
    Used for marking a horizontal or vertical region in plots.
    The region can be dragged and is bounded by lines which can be dragged individually.
    
    ===============================  =============================================================================
    **Signals:**
    sigRegionChangeFinished(self)    Emitted when the user has finished dragging the region (or one of its lines)
                                     and when the region is changed programatically.
    sigRegionChanged(self)           Emitted while the user is dragging the region (or one of its lines)
                                     and when the region is changed programatically.
    ===============================  =============================================================================
    """
    
    sigRegionChangeFinished = QtCore.pyqtSignal(object)
    sigRegionChanged = QtCore.pyqtSignal(object)
    Vertical = 0
    Horizontal = 1
    
    def __init__(self, values=[0, 1], orientation=None, brush=None, pen=None, 
                movable=True, bounds=None, labels=False, textColor=(200,200,200), 
                textBorder=None, textFill=None, textFormat="{:.3f}", unit=None,
                midLine=True, name=None):
        """Create a new LinearRegionItem.
        
        ==============  =====================================================================
        **Arguments:**
        values          A list of the positions of the lines in the region. These are not
                        limits; limits can be set by specifying bounds.
        orientation     Options are InfiniteLineLabel.Vertical or InfiniteLineLabel.Horizontal.
                        If not specified it will be vertical.
        brush           Defines the brush that fills the region. Can be any arguments that
                        are valid for :func:`mkBrush <pyqtgraph.mkBrush>`. Default is
                        trnombreansparent blue.
        pen             Pen to use when drawing lines. Can be any arguments that are valid
                        for :func:`mkPen <pyqtgraph.mkPen>`. Default pen is transparent
                        yellow.
        movable         If True, the region and individual lines are movable by the user; if
                        False, they are static.
        bounds          Optional [min, max] bounding values for the region
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
        midLine         If True, a line located in between the two lines of the item
                        is shown
        name            If not None, correspond to the name of the object
        ==============  =====================================================================
        """
        
        UIGraphicsItem.__init__(self)

        if orientation is None:
            orientation = LinearRegionItem.Vertical
            location = 0.85
            shift = 0.125

        if orientation == LinearRegionItem.Horizontal:
            location = 0.1
            shift = 0.05
        self.orientation = orientation

        self.bounds = bounds

        self.textColor = textColor
        self.location = location
        self.shift = shift
        self.format = textFormat
        self.labels = labels
        self.unit = unit
        self._name = name

        self.anchorLeft = (1., 0.5)
        self.anchorRight = (0., 0.5)
        self.anchorUp = (0.5, 1.)
        self.anchorDown = (0.5, 0.)
        
        self.linear = LinearRegionItemBase(values=values, orientation=orientation,
                         brush=brush, movable=movable, bounds=bounds)
        loc = 0.5*(values[0]+values[1])
        self.line = InfiniteLineBase(pos=loc, movable=False)
        self.line.setPen(fn.mkPen(style=QtCore.Qt.DashLine))
        self.text1 = TextItem(border=textBorder, fill=textFill) # texte de gauche ou du bas
        self.text2 = TextItem(border=textBorder, fill=textFill) # texte de droite ou du haut

        if not self.labels:
            self.text1.hide()
            self.text2.hide()

        if not midLine:
            self.line.hide()

        # important : on s'assurce que les objets sont bien des parents de la classe
        # cela permet de bénéficier de l'ensemble de ses méthodes. 
        self.linear.setParentItem(self)
        self.text1.setParentItem(self)
        self.text2.setParentItem(self)
        self.line.setParentItem(self)

        ## Explicitly wrap methods from LinearRegionItemBase
        for m in ['getRegion', 'setBounds', 'setMovable']:
            setattr(self, m, getattr(self.linear, m))

        # we set the linearRegionItem to be movable or not
        self.setMovable(movable)

        self.linear.sigRegionChanged.connect(self.linearMoved)
        self.linear.sigRegionChangeFinished.connect(self.linearMoveFinished)

    def setRegion(self, rgn):
        """Set the values for the edges of the region.
        
        ==============   ==============================================
        **Arguments:**
        rgn              A list or tuple of the lower and upper values.
        ==============   ==============================================
        """
        self.linear.setRegion(rgn)
        self.update()


    def linearMoved(self):
        self.prepareGeometryChange()
        self.update()
        self.sigRegionChanged.emit(self) 

    def linearMoveFinished(self):
        self.update()
        self.sigRegionChangeFinished.emit(self) 

    def viewRangeChanged(self):
        self.update()

    def boundingRect(self):
        br = UIGraphicsItem.boundingRect(self)
        return br
        
    def paint(self, p, *args):
        pass

    def update(self):
        br = UIGraphicsItem.boundingRect(self)
        xmin, ymin, xmax, ymax = br.getCoords()
        pos1, pos2 = self.linear.getRegion()
        if self.orientation == LinearRegionItem.Vertical:
            posy = ymin+self.location*(ymax-ymin)
            diffMin = pos1-xmin
            diffMax = pos2-xmin
            lim1 = self.shift*(xmax-xmin)
            lim2 = (1.-self.shift)*(xmax-xmin)
            if diffMin < lim1:
                self.text1.anchor = Point(self.anchorRight)
            else:
                self.text1.anchor = Point(self.anchorLeft)
            if diffMax > lim2:
                self.text2.anchor = Point(self.anchorLeft)
            else:
                self.text2.anchor = Point(self.anchorRight)
            fmt = " x = "+self.format 
            if self.unit is not None:
                fmt = fmt + self.unit
            self.text1.setText(fmt.format(pos1), color=self.textColor)
            self.text1.setPos(pos1, posy)
            self.text2.setText(fmt.format(pos2), color=self.textColor)         
            self.text2.setPos(pos2, posy)   
        else:
            posx = xmin+self.location*(xmax-xmin)
            diffMin = pos1-ymin
            diffMax = pos2-ymin
            lim1 = self.shift*(ymax-ymin)
            lim2 = (1.-self.shift)*(ymax-ymin)
            if diffMin < lim1:
                self.text1.anchor = Point(self.anchorUp)
            else:
                self.text1.anchor = Point(self.anchorDown)
            if diffMax > lim2:
                self.text2.anchor = Point(self.anchorDown)
            else:
                self.text2.anchor = Point(self.anchorUp)
            fmt = " x = "+self.format 
            if self.unit is not None:
                fmt = fmt + self.unit
            self.text1.setText(fmt.format(pos1), color=self.textColor)
            self.text1.setPos(posx, pos1)
            self.text2.setText(fmt.format(pos2), color=self.textColor)         
            self.text2.setPos(posx, pos2)
        posss = (pos1+pos2)*0.5
        self.line.setPos(posss)


    def setPen(self, *args, **kwargs):
        """
        Set the pen used to draw the line. Allowable arguments are any that are
        valid for :func:`mkPen <pyqtgraph.mkPen>`.
        """
        for l in self.linear.lines:
            l.setPen(*args, **kwargs)
        # we force the middle line to stay thin and with a specific style
        self.line.pen.setStyle(QtCore.Qt.DashLine)
        self.line.pen.setWidth(1)
        self.update()

    def setHoverPen(self, *args, **kwargs):
        """
        Set the pen used to drawingw the line when the mouse is hovering it.
        Allowable arguments are any that are valid for 
        :func:`mkPen <pyqtgraph.mkPen>`.
        """
        for l in self.linear.lines:
            l.setHoverPen(pen)
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

    def showLabels(self, state):
        """
        Display or not the labels indicating the location of the lines in data
        coordinates.

        ==============   ==============================================
        **Arguments:**
        state            If True, the labels are shown. Otherwise, they
                         are hidden.
        ==============   ==============================================
        """
        if state:
            self.text1.show()
            self.text2.show()
        else:
            self.text1.hide()
            self.text2.hide()
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

    def showMidLine(self, state):
        """
        Show or not the midline.

        ==============   ==============================================
        **Arguments:**
        state             if True, the midline is visible.
        ==============   ==============================================  
        """
        if state:
            self.line.show()
        else:
            self.line.hide()

    def setName(self, name):
        self._name = name
        
    def name(self):
        return self._name