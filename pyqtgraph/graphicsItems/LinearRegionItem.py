from ..Qt import QtGui, QtCore
from .UIGraphicsItem import UIGraphicsItem
from .InfiniteLine import InfiniteLine
from .. import functions as fn
from .. import debug as debug

__all__ = ['LinearRegionItem']

class LinearRegionItem(UIGraphicsItem):
    """
    **Bases:** :class:`UIGraphicsItem <pyqtgraph.UIGraphicsItem>`
    
    Used for marking a horizontal or vertical region in plots.
    The region can be dragged and is bounded by lines which can be dragged individually.
    
    ===============================  ====================================================
    **Signals:**
    sigRegionChangeFinished(self)    Emitted when the user has finished dragging 
                                     the region (or one of its lines)
                                     and when the region is changed programatically.
    sigRegionChanged(self)           Emitted while the user is dragging the region 
                                     (or one of its lines)
                                     and when the region is changed programatically.
    sigRemoveRequested               Emitted when the user selects 'remove' from the 
                                     CustomInfiniteLine's context menu (if available).
    sigBoundsActivated               boolean, Emitted when the bounds are activated via
                                     the context menu
    sigBoundsChanged                 list, Emitted when the bounds are modified via the 
                                     context menu
    sigVisibleText                   boolean, Emitted when the visiblity of the  
                                     textItem has changed
    sigLocationChanged               float, Emitted when the location of the 
                                     TextItem has changed
    sigShiftChanged                  float, Emitted when the shift value has changed
    sigDigitsChanged                 int, emitted when the number of digits of the
                                     label has changed     
    ===============================  ====================================================
    """

    sigRegionChangeFinished = QtCore.Signal(object)
    sigRegionChanged = QtCore.Signal(object)
    sigRemoveRequested = QtCore.Signal(object)
    sigBoundsActivated = QtCore.Signal(object)
    sigBoundsChanged = QtCore.Signal(object)
    sigVisibilityChanged = QtCore.Signal(object)
    sigLocationChanged = QtCore.Signal(object)
    sigShiftChanged = QtCore.Signal(object)
    sigDigitsChanged = QtCore.Signal(object)
    
    Vertical = 0
    Horizontal = 1
    
    def __init__(self, values=[0,1], orientation=None, brush=None, movable=True, 
                 bounds=None, removable=True, visibleLabels=True, shift=0.5, 
                 location=0.05, activateLocation=True,activateShift=True, 
                 nDigits=3, onlyLines=True, visibleMenu=True):
        """
        Create a new LinearRegionItem.
        
        ==============  =====================================================================
        **Arguments:**
        values           A list of the positions of the lines in the region. 
                         These are not limits; limits can be set by specifying 
                         bounds.
        orientation      Options are LinearRegionItem.Vertical or 
                         LinearRegionItem.Horizontal.
                         If not specified it will be vertical.
        brush            Defines the brush that fills the region. Can be any 
                         arguments that are valid for :
                         func:`mkBrush <pyqtgraph.mkBrush>`. Default is
                         transparent blue.
        movable          If True, the region and individual lines are movable 
                         by the user; if False, they are static.
        bounds           Optional [min, max] bounding values for the region
        removable        defines wether or not all the item is removable.
        visibleLabels    make the two labels associated to the InfiniteLine visible       
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
        nDigits          int, number of digits used to format the label
        onlyLines        boolean. If set to True, no labels are visible
        visibleMenu      boolean. Defines if the context menu is visible
        ==============  =====================================================================
        """
        
        UIGraphicsItem.__init__(self)
        if orientation is None:
            orientation = LinearRegionItem.Vertical
        self.orientation = orientation
        self.bounds = QtCore.QRectF()
        self.blockLineSignal = False
        self.moving = False
        self.mouseHovering = False
        self.removable = removable
        
        if orientation == LinearRegionItem.Horizontal:
            self.lines = [
                InfiniteLine(QtCore.QPointF(0, values[0]), 0, 
                             movable=movable, bounds=bounds, 
                             removable=False, visibleMenu=False), 
                InfiniteLine(QtCore.QPointF(0, values[1]), 0, 
                             movable=movable, bounds=bounds, 
                             removable=False, visibleMenu=False)]
        elif orientation == LinearRegionItem.Vertical:
            self.lines = [
                InfiniteLine(QtCore.QPointF(values[1], 0), 90, 
                             movable=movable, bounds=bounds, 
                             removable=False, visibleMenu=False), 
                InfiniteLine(QtCore.QPointF(values[0], 0), 90, 
                             movable=movable, bounds=bounds, 
                             removable=False, visibleMenu=False)]
        else:
            raise Exception('Orientation must be one of LinearRegionItem.Vertical or LinearRegionItem.Horizontal')
        
        
        for l in self.lines:
            l.setParentItem(self)
            l.sigPositionChangeFinished.connect(self.lineMoveFinished)
            l.sigPositionChanged.connect(self.lineMoved)
            
        if brush is None:
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 255, 50))
        self.setBrush(brush)
        
        self.setMovable(movable)
        
        self.useBounds = True
        self.lowerBound, self.upperBound = self.lines[0].maxRange
        if bounds is None:
            self.useBounds = False
            
        self.location = location
        self.shift = shift
        self.visibleLabels = visibleLabels
        self.activateLocation = activateLocation
        self.activateShift = activateShift
        self.nDigits = nDigits
        self.onlyLines = onlyLines
        self.visibleMenu = visibleMenu
        self.menu = LinearRegionItemMenu(self)        
        
    def getRegion(self):
        """Return the values at the edges of the region."""
        #if self.orientation[0] == 'h':
            #r = (self.bounds.top(), self.bounds.bottom())
        #else:
            #r = (self.bounds.left(), self.bounds.right())
        r = [self.lines[0].value(), self.lines[1].value()]
        return (min(r), max(r))

    def setRegion(self, rgn):
        """Set the values for the edges of the region.
        
        ==============   ==============================================
        **Arguments:**
        rgn              A list or tuple of the lower and upper values.
        ==============   ==============================================
        """
        if self.lines[0].value() == rgn[0] and self.lines[1].value() == rgn[1]:
            return
        self.blockLineSignal = True
        self.lines[0].setValue(rgn[0])
        self.blockLineSignal = False
        self.lines[1].setValue(rgn[1])
        #self.blockLineSignal = False
        self.lineMoved()
        self.lineMoveFinished()

    def setBrush(self, *br, **kargs):
        """Set the brush that fills the region. Can have any arguments that are valid
        for :func:`mkBrush <pyqtgraph.mkBrush>`.
        """
        self.brush = fn.mkBrush(*br, **kargs)
        self.currentBrush = self.brush

    def setBounds(self, bounds):
        """Optional [min, max] bounding values for the region. To have no bounds on the
        region use [None, None].
        Does not affect the current position of the region unless it is outside the new bounds. 
        See :func:`setRegion <pyqtgraph.LinearRegionItem.setRegion>` to set the position 
        of the region."""
        for l in self.lines:
            l.setBounds(bounds)
        
    def setMovable(self, m):
        """Set lines to be movable by the user, or not. If lines are movable, they will 
        also accept HoverEvents."""
        for l in self.lines:
            l.setMovable(m)
        self.movable = m
        self.setAcceptHoverEvents(m)

    def boundingRect(self):
        br = UIGraphicsItem.boundingRect(self)
        rng = self.getRegion()
        if self.orientation == LinearRegionItem.Vertical:
            br.setLeft(rng[0])
            br.setRight(rng[1])
        else:
            br.setTop(rng[0])
            br.setBottom(rng[1])
        return br.normalized()
        
    def paint(self, p, *args):
        profiler = debug.Profiler()
        UIGraphicsItem.paint(self, p, *args)
        p.setBrush(self.currentBrush)
        p.setPen(fn.mkPen(None))
        p.drawRect(self.boundingRect())

    def dataBounds(self, axis, frac=1.0, orthoRange=None):
        if axis == self.orientation:
            return self.getRegion()
        else:
            return None

    def lineMoved(self):
        if self.blockLineSignal:
            return
        self.prepareGeometryChange()
        #self.emit(QtCore.SIGNAL('regionChanged'), self)
        self.sigRegionChanged.emit(self)
            
    def lineMoveFinished(self):
        #self.emit(QtCore.SIGNAL('regionChangeFinished'), self)
        self.sigRegionChangeFinished.emit(self)
        
            
    #def updateBounds(self):
        #vb = self.view().viewRect()
        #vals = [self.lines[0].value(), self.lines[1].value()]
        #if self.orientation[0] == 'h':
            #vb.setTop(min(vals))
            #vb.setBottom(max(vals))
        #else:
            #vb.setLeft(min(vals))
            #vb.setRight(max(vals))
        #if vb != self.bounds:
            #self.bounds = vb
            #self.rect.setRect(vb)
        
    #def mousePressEvent(self, ev):
        #if not self.movable:
            #ev.ignore()
            #return
        #for l in self.lines:
            #l.mousePressEvent(ev)  ## pass event to both lines so they move together
        ##if self.movable and ev.button() == QtCore.Qt.LeftButton:
            ##ev.accept()
            ##self.pressDelta = self.mapToParent(ev.pos()) - QtCore.QPointF(*self.p)
        ##else:
            ##ev.ignore()
            
    #def mouseReleaseEvent(self, ev):
        #for l in self.lines:
            #l.mouseReleaseEvent(ev)
            
    #def mouseMoveEvent(self, ev):
        ##print "move", ev.pos()
        #if not self.movable:
            #return
        #self.lines[0].blockSignals(True)  # only want to update once
        #for l in self.lines:
            #l.mouseMoveEvent(ev)
        #self.lines[0].blockSignals(False)
        ##self.setPos(self.mapToParent(ev.pos()) - self.pressDelta)
        ##self.emit(QtCore.SIGNAL('dragged'), self)

    def mouseDragEvent(self, ev):
        if not self.movable or int(ev.button() & QtCore.Qt.LeftButton) == 0:
            return
        ev.accept()
        
        if ev.isStart():
            bdp = ev.buttonDownPos()
            self.cursorOffsets = [l.pos() - bdp for l in self.lines]
            self.startPositions = [l.pos() for l in self.lines]
            self.moving = True
            
        if not self.moving:
            return
            
        #delta = ev.pos() - ev.lastPos()
        self.lines[0].blockSignals(True)  # only want to update once
        for i, l in enumerate(self.lines):
            l.setPos(self.cursorOffsets[i] + ev.pos())
            #l.setPos(l.pos()+delta)
            #l.mouseDragEvent(ev)
        self.lines[0].blockSignals(False)
        self.prepareGeometryChange()
        
        if ev.isFinish():
            self.moving = False
            self.sigRegionChangeFinished.emit(self)
        else:
            self.sigRegionChanged.emit(self)
            
    def mouseClickEvent(self, ev):
        if self.moving and ev.button() == QtCore.Qt.RightButton:
            ev.accept()
            for i, l in enumerate(self.lines):
                l.setPos(self.startPositions[i])
            self.moving = False
            self.sigRegionChanged.emit(self)
            self.sigRegionChangeFinished.emit(self)
        if ev.button() == QtCore.Qt.RightButton and self.contextMenuEnabled():
            self.raiseContextMenu(ev)
            ev.accept()

    def contextMenuEnabled(self):      
        return self.visibleMenu    

    def raiseContextMenu(self, ev):
        if not self.contextMenuEnabled():
            return
        menu = self.getMenu()
        menu = self.scene().addParentContextMenus(self, menu, ev)
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(pos.x(), pos.y()))  
        
    def getMenu(self):
        return self.menu          
        
    def setLowerBound(self,lowerBound):
        self.lowerBound = lowerBound
        
    def setUpperBound(self,upperBound):
        self.upperBound = upperBound         

    def hoverEvent(self, ev):
        if self.movable and (not ev.isExit()) and ev.acceptDrags(QtCore.Qt.LeftButton):
            self.setMouseHover(True)
        else:
            self.setMouseHover(False)
            
    def setMouseHover(self, hover):
        ## Inform the item that the mouse is(not) hovering over it
        if self.mouseHovering == hover:
            return
        self.mouseHovering = hover
        if hover:
            c = self.brush.color()
            c.setAlpha(c.alpha() * 2)
            self.currentBrush = fn.mkBrush(c)
        else:
            self.currentBrush = self.brush
        self.update()

    #def hoverEnterEvent(self, ev):
        #print "rgn hover enter"
        #ev.ignore()
        #self.updateHoverBrush()

    #def hoverMoveEvent(self, ev):
        #print "rgn hover move"
        #ev.ignore()
        #self.updateHoverBrush()

    #def hoverLeaveEvent(self, ev):
        #print "rgn hover leave"
        #ev.ignore()
        #self.updateHoverBrush(False)
        
    #def updateHoverBrush(self, hover=None):
        #if hover is None:
            #scene = self.scene()
            #hover = scene.claimEvent(self, QtCore.Qt.LeftButton, scene.Drag)
        
        #if hover:
            #self.currentBrush = fn.mkBrush(255, 0,0,100)
        #else:
            #self.currentBrush = self.brush
        #self.update()

class LinearRegionItemMenu(QtGui.QMenu):    
    
    def __init__(self, linearRegionItem):
        QtGui.QMenu.__init__(self)
        self.linearRegionItem = linearRegionItem
        if self.linearRegionItem.removable:
            remAct = QtGui.QAction("Remove item", self)
            remAct.triggered.connect(self.removeClicked)
            self.addAction(remAct)
        
        self.editLine = QtGui.QMenu("Edit item")
        groupLine = QtGui.QWidgetAction(self)
        widgetLine = self.createWidget()
        groupLine.setDefaultWidget(widgetLine)
        self.editLine.addAction(groupLine)
        self.addMenu(self.editLine)        
        
        self.cbBounds.toggled.connect(self.updateStateBounds)
        self.ledtMin.textChanged.connect(self.updateMinValue)
        self.ledtMax.textChanged.connect(self.updateMaxValue)
        self.cbText.toggled.connect(self.updateStateText)
        self.sldLocation.valueChanged.connect(self.updateLocation)
        self.sldShift.valueChanged.connect(self.updateShift)
        self.ledtDigits.textChanged.connect(self.updateDigits)

    def createWidget(self):
        self.cbBounds = QtGui.QGroupBox("Activate bounds")
        self.cbBounds.setCheckable(True)
        self.lblMin = QtGui.QLabel("min value")
        self.lblMax = QtGui.QLabel("max value") 
        self.ledtMin = QtGui.QLineEdit()
        tip = "defines the lower bound of the line. If the line is empty\n"
        tip += "or if the input can not be converted into a float, the value\n"
        tip += "of the bound is set to 'None' and the bound is not active."
        tip += "This bound is the same for the two lines."
        self.ledtMin.setToolTip(tip)        
        self.ledtMin.setFixedWidth(70)
        self.ledtMax = QtGui.QLineEdit()
        self.ledtMax.setFixedWidth(70)
        tip = "defines the upper bound of the line. If the line is empty\n"
        tip += "or if the input can not be converted into a float, the value\n"
        tip += "of the bound is set to 'None' and the bound is not active."
        tip += "This bound is the same for the two lines."        
        self.ledtMax.setToolTip(tip)        
        
        if self.linearRegionItem.useBounds :        
            valmin = self.linearRegionItem.lowerBound
            valmax = self.linearRegionItem.upperBound
            self.cbBounds.setChecked(True)
            if valmin is not None:
                self.ledtMin.setText(str(valmin))
            if valmax is not None:
                self.ledtMax.setText(str(valmax))
        else:
            self.cbBounds.setChecked(False)

        gridLayout = QtGui.QGridLayout()        
        gridLayout.addWidget(self.lblMin,0,0)
        gridLayout.addWidget(self.ledtMin,0,1)
        gridLayout.addWidget(self.lblMax,1,0)
        gridLayout.addWidget(self.ledtMax,1,1) 
        self.cbBounds.setLayout(gridLayout)   

        self.cbText = QtGui.QGroupBox("Visible label")
        self.cbText.setCheckable(True)
        self.lblLocation = QtGui.QLabel("location")
        self.lblShift = QtGui.QLabel("shift") 
        self.sldLocation = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.sldLocation.setTickPosition(QtGui.QSlider.TicksBelow)
        self.sldLocation.setTickInterval(100)
        tip = "defines the location of the label with respect to the\n"
        tip += "two opposite axis. If the value is set to 0, the\n"
        tip += "label will be located at the left (or bottom axis).\n"
        tip += "If the value is set to 1, the label will be located\n"
        tip += "at the opposite axis. Otherwise, it will be located\n"
        tip += "in between."
        self.sldLocation.setToolTip(tip)        
        self.sldLocation.setRange(0,1000)
        self.sldLocation.setValue(int(1000*round(self.linearRegionItem.location,3)))
        self.sldLocation.setFixedWidth(70)
        self.sldShift = QtGui.QSlider(QtCore.Qt.Horizontal)  
        self.sldShift.setTickPosition(QtGui.QSlider.TicksBelow)
        self.sldShift.setTickInterval(100)
        tip = "defines the shift of the label with respect to the\n"
        tip += "line depending on the location of the line on the\n."
        tip += "axis. If the value is set to 0, the label will always\n"
        tip += "be located above the line. If the value is set to 1,\n"
        tip += "the label will always be located below the line."        
        self.sldShift.setToolTip(tip)        
        self.sldShift.setRange(0,1000)
        self.sldShift.setValue(int(1000*round(self.linearRegionItem.shift,3))) 
        self.sldShift.setFixedWidth(70)
        if not self.linearRegionItem.activateLocation:
            self.lblLocation.setEnabled(False)
            self.sldLocation.setEnabled(False)
        if not self.linearRegionItem.activateShift:
            self.lblShift.setEnabled(False)
            self.sldShift.setEnabled(False)            
        if self.linearRegionItem.visibleLabels :
            self.cbText.setChecked(True)
            self.sldLocation.setValue(int(1000*self.linearRegionItem.location))
            self.sldShift.setValue(int(1000*self.linearRegionItem.shift))
        self.lblDigits = QtGui.QLabel("digits")
        self.ledtDigits = QtGui.QLineEdit()
        self.ledtDigits.setValidator(QtGui.QIntValidator())
        self.ledtDigits.setText(str(self.linearRegionItem.nDigits))
        self.ledtDigits.setFixedWidth(70)
        tip = "defines the number of digits of the label"
        self.ledtDigits.setToolTip(tip)
        
        gridLayout = QtGui.QGridLayout()
        gridLayout.addWidget(self.lblLocation, 0, 0)
        gridLayout.addWidget(self.sldLocation, 0, 1)
        gridLayout.addWidget(self.lblShift, 1, 0)
        gridLayout.addWidget(self.sldShift, 1, 1)
        gridLayout.addWidget(self.lblDigits, 2, 0)
        gridLayout.addWidget(self.ledtDigits, 2, 1)
        self.cbText.setLayout(gridLayout)

        widget = QtGui.QWidget()
        vBox = QtGui.QVBoxLayout()
        vBox.addWidget(self.cbBounds)
        vBox.addWidget(self.cbText)
        widget.setLayout(vBox)
        widget.setContentsMargins(0, 0, 0, 0)
        widget.layout().setContentsMargins(0, 0, 0, 0)   
        
        # if we only display the line, we hide the label and the specific part 
        # of the contextMenu related to the label
        if self.linearRegionItem.onlyLines:
            self.linearRegionItem.visibleLabels = False
            QtCore.QTimer.singleShot(0, lambda: self.linearRegionItem.sigVisibilityChanged.emit(False))
            self.cbText.hide()          
        
        return widget
        
    def updateLocation(self,value):
        floatValue = fn.remapSlider(value, 0, 1000)
        self.linearRegionItem.location = floatValue        
        self.linearRegionItem.sigLocationChanged.emit(floatValue)
        
    def updateShift(self,value):
        floatValue = fn.remapSlider(value, 0, 1000)
        self.linearRegionItem.shift = floatValue
        self.linearRegionItem.sigShiftChanged.emit(floatValue)

    def updateStateText(self,state):
        self.linearRegionItem.visibleLabels = state
        #self.enableTextWidgets(state)
        if state and not self.linearRegionItem.activateLocation:
            self.lblLocation.setEnabled(False)
            self.sldLocation.setEnabled(False)
        if state and not self.linearRegionItem.activateShift:
            self.lblShift.setEnabled(False)
            self.sldShift.setEnabled(False)             
        self.linearRegionItem.sigVisibilityChanged.emit(state)
        
    def updateStateBounds(self,state):
        bound = [None, None]
        if state:
            bound = [self.linearRegionItem.lowerBound,self.linearRegionItem.upperBound]
        self.linearRegionItem.useBounds = state
        for line in self.linearRegionItem.lines:
            line.setBounds(bound)
        self.linearRegionItem.setBounds(bound) 
        self.linearRegionItem.sigBoundsActivated.emit(state)
        self.linearRegionItem.sigBoundsChanged.emit(bound)    
        
    def updateMinValue(self,value):        
        try:
            minval = float(value)
        except:
            minval = None
        self.linearRegionItem.lowerBound = minval  
        for line in self.linearRegionItem.lines:
            currentPos = line.value()
            if minval is not None and currentPos < minval:
                line.setPos(minval)
        self.updateBounds()
            
    def updateMaxValue(self,value):
        try:
            maxval = float(value)           
        except:
            maxval = None
        self.linearRegionItem.upperBound = maxval
        for line in self.linearRegionItem.lines:
            currentPos = line.value()
            if maxval is not None and currentPos > maxval:
                line.setPos(maxval)
        self.updateBounds() 
        
    def updateDigits(self):
        digits = self.ledtDigits.text()
        if digits == "":
            return
        else:
            self.linearRegionItem.sigDigitsChanged.emit(int(digits))        
        
    def updateBounds(self):
        lowerBound = self.linearRegionItem.lowerBound
        upperBound = self.linearRegionItem.upperBound
        for line in self.linearRegionItem.lines:
            line.setBounds([lowerBound,upperBound])
        #self.linearRegionItem.setBounds([lowerBound,upperBound])
        self.linearRegionItem.sigBoundsChanged.emit([lowerBound,upperBound])

    def removeClicked(self):
        ## Send remove event only after we have exited the menu event handler
        QtCore.QTimer.singleShot(0, lambda: self.linearRegionItem.sigRemoveRequested.emit(self))
  
