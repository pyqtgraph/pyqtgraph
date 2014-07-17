# -*- coding: utf-8 -*-
"""
Class used to add (or remove) a Crosshair Item with a pg.TextItem 
indicating the value of the location on a PlotItem

This item is based on crosshair/mouse interaction example and
adds some extra functionalities

@author : Vincent Le Saux (vincent.le_saux@ensta-bretagne.fr)
"""

from .InfiniteLine import InfiniteLine
from .TextItem import TextItem
from ..Point import Point

class CrosshairLabel(object):

    def __init__(self,item, posX=None,posY=None,pen=None,movable=False,vbounds=None,
                 hbounds=None,nDigits=3,removable=True,visibleLabel=True,fill=None,
                 textColor=None,visibleMenu=True):
        """
        =============== =======================================================
        **Arguments:**
        item            a PlotItem object
        posX            float, X location of the crosshair item
        posY            float, Y location of the crosshair item
        pen             Pen to use when drawing line. Can be any arguments that
                        are valid for :func:`mkPen <pyqtgraph.mkPen>`. Default 
                        pen is transparent blue (when set to None).
        movable         If True, the line can be dragged to a new position by 
                        the user.
        vbounds         Optional [min, max] bounding values for the vertical
                        line
        hbounds         Optional [min, max] bounding values for the horizontal 
                        line
        ndigits         int, number of digits used to format the label
        removable       If True, the object can be remove via a contextMenu 
                        activated by right-clicking on the object
        visibleLabel    make the label associated to the crosshair item visible
        fill            color of the TextItem background        
        textColor       color used for the textItem
        shift           float (must be in the [0,1] interval) used to switch 
                        the TextItems from one side of the axis to the other 
                        in order to increase its visibility
        visibleMenu     boolean. Defines is the contextMenu is enabled or not
        =============== ======================================================= 
        """
        self.cilHor = InfiniteLine(pos=posX, angle=0, movable=movable,
                                   pen=pen, bounds=hbounds, removable=removable,
                                   onlyLine=True) 
        self.cilVer = InfiniteLine(pos=posY, angle=90, movable=movable,
                                   pen=pen, bounds=vbounds, removable=removable,
                                   onlyLine=True) 
        self.item = item
        self.shift = 0.5
        self.visibleLabel = visibleLabel
        self.nDigits = nDigits
        self.fill = fill
        if textColor is None:
            textColor = (0,0,255)
        self.textColor = textColor
        self.anchorTopLeft = (1.,1.)
        self.anchorTopRight = (0.,1.)
        self.anchorBottomLeft = (1.,0.)
        self.anchorBottomRight = (0.,0.)
        self.moving = False
        self.posValue = TextItem(text="",anchor=self.anchorTopLeft,fill=self.fill)
        self.posValue.setPos(0,0)
        self.addItems(self.posValue, self.cilHor, self.cilVer)    
        self.update()  
        
        self.cilHor.sigPositionChanged.connect(self.isMoving) 
        self.cilHor.sigPositionChangeFinished.connect(self.stopMoving)
        self.cilVer.sigPositionChanged.connect(self.isMoving) 
        self.cilVer.sigPositionChangeFinished.connect(self.stopMoving)        
        self.item.getViewBox().sigYRangeChanged.connect(self.update)
        self.item.getViewBox().sigXRangeChanged.connect(self.update)
        self.cilHor.sigRemoveRequested.connect(self.removeItems)
        self.cilVer.sigRemoveRequested.connect(self.removeItems)
        self.item.scene().sigMouseMoved.connect(self.updatePosition)
        
    def isMoving(self):
        self.moving = True
        
    def stopMoving(self):
        self.moving = False
        
    def updatePosition(self, pos):
        if not self.moving:
            return
        if self.item.sceneBoundingRect().contains(pos):
            mousePoint = self.item.getViewBox().mapSceneToView(pos)
            self.cilHor.setValue(mousePoint.y())
            self.cilVer.setValue(mousePoint.x())
            self.update()
            
    def update(self):
        posY = self.cilHor.value()
        posX = self.cilVer.value()
        limits = self.item.getViewBox().viewRange() 
        xmin, xmax = limits[0] 
        ymin, ymax = limits[1]
        diffX = xmax-xmin
        diffY = ymax-ymin
        diffXMin = posX-xmin
        diffYMin = posY-ymin
        limX = self.shift*diffX
        limY = self.shift*diffY
        if diffXMin < limX:
            if diffYMin < limY:
                self.posValue.anchor = Point(self.anchorTopRight)
            else:
                self.posValue.anchor = Point(self.anchorBottomRight)
        else:
            if diffYMin < limY:
                self.posValue.anchor = Point(self.anchorTopLeft)
            else:
                self.posValue.anchor = Point(self.anchorBottomLeft)
        fmt = " x = {:."+str(self.nDigits)+"f} \n y = {:."+str(self.nDigits)+"f} "
        self.posValue.setText(fmt.format(posX, posY), color=self.textColor)
        self.posValue.setPos(posX, posY)

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
        self.item.removeItem(self.cilHor)
        self.item.removeItem(self.cilVer)
        self.item.removeItem(self.posValue)                    
