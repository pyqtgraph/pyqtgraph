# -*- coding: utf-8 -*-
"""
Class used to add (or remove) a LinearRegionItem combined with two
pg.TextItem indicating the value of the two InfiniteLine that delimitates
the LinearRegionItem region to a PlotItem

@author : Vincent Le Saux (vincent.le_saux@ensta-bretagne.fr)
"""

from .LinearRegionItem import LinearRegionItem
from .TextItem import TextItem
from ..Point import Point

__all__ = ["LinearRegionItemLabel"]

class LinearRegionItemLabel(object):

    def __init__(self,item,values=[0,1], orientation=None, brush=None,
                 movable=True, bounds=None, nDigits=3, removable=True,
                 visibleLabels=True, fill=None, textColor=None, location=0.75,
                 shift=0.125, activateLocation=True, activateShift=True,
                 onlyLines=True):
        """
        =============== =======================================================
        **Arguments:**
        
        item             a PlotItem object
        values           the location by defult of the two InfiniteLines that
                         delimitates the LinearRegionItem
        orientationthe   orientation of the InfiniteLines. Must be of the type
                         pg.LinearRegionItem.Horizontal or 
                         pg.LinearRegionItem.Vertical or 
                         None
        brush            the fill pattern between the two InfiniteLines
        movable          set to True if the LinearRegionItem is draggable
        bounds           defines some limits to the LinearRegionItem displacement
                         such as the item can not go beyond these values
        nDigits          int, number of term after the dot                         
        removable        defines wether or not all the items added to the item
                         are removable. If set to True, a contextMenu appears
                         while right-clicking on the object
        visibleLabels    make the two labels associated to the InfiniteLine visible
        fill             color of the TextItems background
        textColor        color used for the textItem         
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
        =============== =======================================================      
        """
        self.clr = LinearRegionItem(values=values, orientation=orientation, 
                                          brush=brush, movable=movable, bounds=bounds, 
                                          removable=removable,visibleLabels=visibleLabels,
                                          location=location,shift=shift,
                                          activateLocation=activateLocation,
                                          activateShift=activateShift,
                                          onlyLines=onlyLines)
        self.item = item
        if orientation == LinearRegionItem.Horizontal:
            location = 0.10
            shift = 0.05
        self.fill = fill
        self.location = location
        self.shift = shift
        self.orientation = orientation
        self.visibleLabels = visibleLabels
        self.anchorLeft = (1.,0.5)
        self.anchorRight = (0.,0.5)
        self.anchorUp = (0.5,1.)
        self.anchorDown = (0.5,0.)
        self.nDigits = nDigits
        self.visibleLabels = visibleLabels
        self.fill = fill
        if textColor is None:
            textColor = (0, 0, 255)
        self.textColor = textColor
        self.lowerLimit = TextItem(text="",anchor=self.anchorLeft,fill=self.fill)
        self.upperLimit = TextItem(text="",anchor=self.anchorRight, fill=self.fill)
        self.addItems(self.lowerLimit,self.upperLimit,self.clr)    
        self.update()
        
        self.clr.sigRegionChanged.connect(self.update)
        self.item.getViewBox().sigYRangeChanged.connect(self.update)
        self.item.getViewBox().sigXRangeChanged.connect(self.update)
        self.clr.sigRemoveRequested.connect(self.removeItems)
        self.clr.sigVisibilityChanged.connect(self.updateVisibilityTexts)
        self.clr.sigLocationChanged.connect(self.updateLocationValue)
        self.clr.sigShiftChanged.connect(self.updateShiftValue)
        self.clr.sigDigitsChanged.connect(self.updateDigits)

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
        self.item.removeItem(self.clr)
        self.item.removeItem(self.lowerLimit)
        self.item.removeItem(self.upperLimit)

    def updateVisibilityTexts(self, state):
        self.visibleLabels = state
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
        """
        Update the location of the TextItems depending on the location of the
        LinearRegionItems and the XAxis and YAxis range
        """
        limits = self.item.getViewBox().viewRange()
        xmin, xmax = limits[0]
        ymin, ymax = limits[1]
        clrLower, clrUpper = self.clr.getRegion()
        if self.orientation == LinearRegionItem.Vertical:
            pos = ymin+self.location*(ymax-ymin)
            diffX = xmax-xmin
            diffMin = clrLower-xmin
            limInf = self.shift*diffX
            limSup = (1.-self.shift)*diffX
            diffMax = clrUpper-xmin
            self.lowerLimit.anchor = Point(self.anchorLeft)
            self.upperLimit.anchor = Point(self.anchorRight)
            if diffMin < limInf:
                self.lowerLimit.anchor = Point(self.anchorRight)
            if diffMax > limSup:
                self.upperLimit.anchor = Point(self.anchorLeft)
            fmt = " x = {:."+str(self.nDigits)+"f}"
            self.lowerLimit.setText(fmt.format(clrLower), color=self.textColor)
            self.lowerLimit.setPos(clrLower, pos)
            self.upperLimit.setText(fmt.format(clrUpper), color=self.textColor)         
            self.upperLimit.setPos(clrUpper, pos)
        else:
            pos = xmin+self.location*(xmax-xmin)
            diffY = ymax-ymin
            diffMin = clrLower-ymin
            limInf = self.shift*diffY
            limSup = (1.-self.shift)*diffY
            diffMax = clrUpper-ymin
            self.lowerLimit.anchor = Point(self.anchorDown)
            self.upperLimit.anchor = Point(self.anchorUp)
            if diffMin < limInf:
                self.lowerLimit.anchor = Point(self.anchorUp)
            if diffMax > limSup:
                self.upperLimit.anchor = Point(self.anchorDown)
            fmt = " x = {:."+str(self.nDigits)+"f}"
            self.lowerLimit.setText(fmt.format(clrLower), color=self.textColor)
            self.lowerLimit.setPos(pos,clrLower)
            self.upperLimit.setText(fmt.format(clrUpper), color=self.textColor)         
            self.upperLimit.setPos(pos,clrUpper)
        self.showLabels(self.visibleLabels)

    def setLocation(self, location):
        """ Modify the x or y location of the TextItem """
        if not 0 < location < 1:
            return
        self.location = location
        self.update()

    def setShift(self, shift):
        """ Modify the valueof the shift used for the location of the textItems """
        if not 0 < shift < 1:
            return
        self.shift = shift
        self.update()

    def showLabels(self, state):
        if state:
            self.lowerLimit.show()
            self.upperLimit.show()
        else:
            self.lowerLimit.hide()
            self.upperLimit.hide()
