# -*- coding: utf-8 -*-
"""
Class used to add (or remove) an InfiniteLine with a pg.TextItem 
indicating the value of the InfiniteLine to a PlotItem. Some 
additionnal actions are also provided (display or not the label
and change some changes)

@author : Vincent Le Saux (vincent.le_saux@ensta-bretagne.fr)
"""
from .InfiniteLine import InfiniteLine
from .TextItem import TextItem
from ..Point import Point

__all__ = ["InfiniteLineLabel"]

class InfiniteLineLabel(object):

    def __init__(self,item, pos=None, angle=90, pen=None, movable=False,
                 bounds=None, nDigits=3, removable=True, visibleLabel=True,
                 fill=None, textColor=None, location=0.05, shift=0.5, 
                 activateLocation=True, activateShift=True, onlyLine=False,
                 visibleMenu=True):
        """
        =============== =======================================================
        **Arguments:**
        
        item             a PlotItem object
        pos              the location by default of the InfiniteLine
        angle            angle of line in degrees. 0 is horizontal, 90 is vertical.
        pen              the pen used to draw the line
        movable          set to True if the LinearRegionItem is draggable
        bounds           defines some limits to the InfiniteLine displacement
                         such as the item can not go beyond these values
        removable        defines wether or not all the items added to the item
                         are removable. If set to True, a contextMenu appears
                         while right-clicking on the object
        visibleLabel     make the label associated to the InfiniteLine visible
        fill             color of the TextItem background       
        textColor        color used for the textItem 
        nDigits          int, number digits used to format the label     
        location         float (must be in the [0,1] interval) used to specify
                         the location of the TextItems
                         value = 0 -> located at the lower axis
                         value = 1 -> located at the upper axis
        shift            float (must be in in the [0,1] interval) used to switch 
                         the TextItems from one side of the axis to the other 
                         in order to increase its visibility
        activateLocation boolean used to activate of not the possibility to
                         modify the location parameters from the context menu
        activateShift    boolean used to activate or not the possibility to
                         modify the shift parameters from the context menu
        visibleMenu      boolean. Defines if the context menu is visible
        onlyLine         boolean. Defines if only the line is visible, not its
                         label (
        =============== =======================================================
        """
        self.cil = InfiniteLine(pos=pos, angle=angle, movable=movable,
                                pen=pen, bounds=bounds, removable=removable,
                                visibleLabel=visibleLabel, location=location,
                                shift=shift, activateShift=activateShift,
                                onlyLine=onlyLine, nDigits=nDigits,
                                visibleMenu=visibleMenu)
        self.item = item
        if angle == 0:
            location = 0.065
        self.location = location
        self.shift = shift
        self.angle = angle
        self.nDigits = nDigits
        self.visibleLabel = visibleLabel
        self.fill = fill
        if textColor is None:
            textColor = (0, 0, 255)
        self.textColor = textColor
        self.anchorLeft = (1., 0.5)
        self.anchorRight = (0., 0.5)
        self.anchorUp = (0.5, 1.)
        self.anchorDown = (0.5, 0.)
        self.posValue = TextItem(text="", anchor=self.anchorLeft,
                                 fill=self.fill)
        self.posValue.setPos(0, 0)
        self.addItems(self.cil, self.posValue)

        self.update()

        self.cil.sigPositionChanged.connect(self.update)
        self.item.getViewBox().sigYRangeChanged.connect(self.update)
        self.item.getViewBox().sigXRangeChanged.connect(self.update)
        self.cil.sigRemoveRequested.connect(self.removeItems)
        self.cil.sigVisibilityChanged.connect(self.updateVisibilityText)
        self.cil.sigLocationChanged.connect(self.updateLocationValue)
        self.cil.sigShiftChanged.connect(self.updateShiftValue)
        self.cil.sigDigitsChanged.connect(self.updateDigits)

    def addItems(self, *args):
        """
        Add the items to the PlotItem instance
        """
        for item in args:
            self.item.addItem(item)

    def removeItems(self):
        """
        Remove the items from the PlotItem instance
        """
        self.item.removeItem(self.cil)
        self.item.removeItem(self.posValue)

    def updateVisibilityText(self, state):
        self.visibleLabel = state
        self.update()

    def updateLocationValue(self, value):
        self.location = value
        self.update()

    def updateShiftValue(self, value):
        self.shift = value
        self.update()

    def updateDigits(self, value):
        self.nDigits = value
        self.update()

    def update(self):
        pos = self.cil.value()
        limits = self.item.getViewBox().viewRange()
        xmin, xmax = limits[0]
        ymin, ymax = limits[1]
        if self.angle == 90:
            diffX = xmax-xmin
            diffMin = pos-xmin
            limInf = self.shift*diffX
            ypos = ymin + self.location*(ymax-ymin)
            self.posValue.anchor = Point(self.anchorLeft)
            if diffMin < limInf:
                self.posValue.anchor = Point(self.anchorRight)
            fmt = " x = {:."+str(self.nDigits)+"f}"
            self.posValue.setText(fmt.format(pos), color=self.textColor)
            self.posValue.setPos(pos, ypos)
        else:
            diffY = ymax-ymin
            diffMin = pos-ymin
            limInf = self.shift*diffY
            xpos = xmin+self.location*(xmax-xmin)
            self.posValue.anchor = Point(self.anchorDown)
            if diffMin < limInf:
                self.posValue.anchor = Point(self.anchorUp)
            fmt = " y = {:."+str(self.nDigits)+"f}"
            self.posValue.setText(fmt.format(pos), color=self.textColor)
            self.posValue.setPos(xpos, pos)
        self.showLabel(self.visibleLabel)

    def setLocation(self, location):
        if not 0 < location < 1:
            return
        self.location = location
        self.update()

    def setShift(self, shift):
        if not 0 < shift < 1:
            return
        self.shift = shift
        self.update()

    def showLabel(self, state):
        if state:
            self.posValue.show()
        else:
            self.posValue.hide()
      
