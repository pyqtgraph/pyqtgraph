from ..Qt import QtGui, QtCore
from ..Point import Point
from .GraphicsObject import GraphicsObject
from .. import functions as fn


__all__ = ['InfiniteLine']


class InfiniteLine(GraphicsObject):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`

    Displays a line of infinite length.
    This line may be dragged to indicate a position in data coordinates.

    =============================== ===================================================
    **Signals:**
    sigDragged
    sigPositionChangeFinished
    sigPositionChanged
    sigRemoveRequested          Emitted when the user selects 'remove' from the
                                CustomInfiniteLine's context menu (if available)
    sigBoundsActivated          boolean, emitted when the bounds are activated
                                via the context menu
    sigBoundsChanged            list, emitted when the bounds are modified via
                                the context menu
    sigVisibilityChanged        boolean, emitted when the visiblity of the
                                textItem has changed
    sigLocationChanged          float, emitted when the location of the TextItem
                                has changed
    sigShiftChanged             float, emitted when the shift value has changed
    sigDigitsChanged            int, emitted when the number of digits of the
                                label has changed    
    =============================== ===================================================
    """

    sigDragged = QtCore.Signal(object)
    sigPositionChangeFinished = QtCore.Signal(object)
    sigPositionChanged = QtCore.Signal(object)
    sigRemoveRequested = QtCore.Signal(object)
    sigBoundsActivated = QtCore.Signal(object)
    sigBoundsChanged = QtCore.Signal(object)
    sigVisibilityChanged = QtCore.Signal(object)
    sigLocationChanged = QtCore.Signal(object)
    sigShiftChanged = QtCore.Signal(object)
    sigDigitsChanged = QtCore.Signal(object)

    def __init__(self, pos=None, angle=90, pen=None, movable=False, bounds=None,
                 removable=True, visibleLabel=True, shift=0.5, location=0.05,
                 activateLocation=True, activateShift=True, nDigits=3,
                 onlyLine=True, visibleMenu=True):
        """
        =============== =======================================================
        **Arguments:**
        pos              Position of the line. This can be a QPointF or a
                         single value for vertical/horizontal lines.
        angle            Angle of line in degrees. 0 is horizontal, 90 is
                         vertical.
        pen              Pen to use when drawing line. Can be any arguments
                         that are valid for :func:`mkPen <pyqtgraph.mkPen>`.
                         Default pen is transparent blue (when set to None).
        movable          If True, the line can be dragged to a new position by
                         the user.
        bounds           Optional [min, max] bounding values. Bounds are only
                         valid if the line is vertical or horizontal. If a
                         bound is not active, its value is set to None
        removable        If True, the object can be remove via a contextMenu
                         activated by right-clicking on the object
        visibleLabel     make the label associated to the InfiniteLine visible
        location         float (must be in the [0,1] interval) used to specify
                         the location of the TextItems
                         value = 0 -> located at the lower axis
                         value = 1 -> located at the upper axis
        shift            float (must be in in the [0,1] interval) used to
                         switch the TextItems from one side of the line to the
                         other in order to increase its visibility
        activateLocation boolean used to activate of not the possibility to
                         modify the location parameters from the context menu
        activateShift    boolean used to activate or not the possibility to
                         modify the shift parameters from the context menu
        nDigits          int, number of digits used to format the label
        onlyLine         boolean, defines if the line acts as a single Infinite
                         Line or as a part of a more complex object (with a 
                         label, etc.). 
        visibleMenu      boolean, defines if the context menu is active or not
        =============== =======================================================
        """
        GraphicsObject.__init__(self)

        if bounds is None:   ## allowed value boundaries for orthogonal lines
            self.maxRange = [None, None]
        else:
            self.maxRange = bounds
        self.moving = False
        self.setMovable(movable)
        self.mouseHovering = False
        self.p = [0, 0]
        self.setAngle(angle)
        if pos is None:
            pos = Point(0, 0)
        self.setPos(pos)

        if pen is None:
            pen = (0, 0, 255)#(200, 200, 100)
        
        self.setPen(pen)
        self.setHoverPen(color=(0, 0, 255), width=2.*self.pen.width())
        self.currentPen = self.pen
        #self.setFlag(self.ItemSendsScenePositionChanges)
        
        self.removable = removable
        self.visibleLabel = visibleLabel        
        self.useBounds = True
        self.lowerBound, self.upperBound = self.maxRange
        if bounds is None:
            self.useBounds = False
        self.location = location
        self.shift = shift
        self.activateLocation = activateLocation
        self.activateShift = activateShift
        self.onlyLine = onlyLine
        self.nDigits = nDigits
        self.visibleMenu = visibleMenu
        self.menu = InfiniteLineMenu(self)
      
    def setMovable(self, m):
        """Set whether the line is movable by the user."""
        self.movable = m
        self.setAcceptHoverEvents(m)
      
    def setBounds(self, bounds):
        """Set the (minimum, maximum) allowable values when dragging."""
        self.maxRange = bounds
        self.setValue(self.value())
        
    def setPen(self, *args, **kwargs):
        """Set the pen for drawing the line. Allowable arguments are any that are valid 
        for :func:`mkPen <pyqtgraph.mkPen>`."""
        self.pen = fn.mkPen(*args, **kwargs)
        if not self.mouseHovering:
            self.currentPen = self.pen
            self.update()
        
    def setHoverPen(self, *args, **kwargs):
        """Set the pen for drawing the line while the mouse hovers over it. 
        Allowable arguments are any that are valid 
        for :func:`mkPen <pyqtgraph.mkPen>`.
        
        If the line is not movable, then hovering is also disabled.
        
        Added in version 0.9.9."""
        self.hoverPen = fn.mkPen(*args, **kwargs)
        if self.mouseHovering:
            self.currentPen = self.hoverPen
            self.update()
        
    def setAngle(self, angle):
        """
        Takes angle argument in degrees.
        0 is horizontal; 90 is vertical.
        
        Note that the use of value() and setValue() changes if the line is 
        not vertical or horizontal.
        """
        self.angle = ((angle+45) % 180) - 45   ##  -45 <= angle < 135
        self.resetTransform()
        self.rotate(self.angle)
        self.update()
        
    def setPos(self, pos):
        
        if type(pos) in [list, tuple]:
            newPos = pos
        elif isinstance(pos, QtCore.QPointF):
            newPos = [pos.x(), pos.y()]
        else:
            if self.angle == 90:
                newPos = [pos, 0]
            elif self.angle == 0:
                newPos = [0, pos]
            else:
                raise Exception("Must specify 2D coordinate for non-orthogonal lines.")
            
        ## check bounds (only works for orthogonal lines)
        if self.angle == 90:
            if self.maxRange[0] is not None:    
                newPos[0] = max(newPos[0], self.maxRange[0])
            if self.maxRange[1] is not None:
                newPos[0] = min(newPos[0], self.maxRange[1])
        elif self.angle == 0:
            if self.maxRange[0] is not None:
                newPos[1] = max(newPos[1], self.maxRange[0])
            if self.maxRange[1] is not None:
                newPos[1] = min(newPos[1], self.maxRange[1])
            
        if self.p != newPos:
            self.p = newPos
            GraphicsObject.setPos(self, Point(self.p))
            self.update()
            self.sigPositionChanged.emit(self)

    def getXPos(self):
        return self.p[0]
        
    def getYPos(self):
        return self.p[1]
        
    def getPos(self):
        return self.p

    def value(self):
        """Return the value of the line. Will be a single number for horizontal and 
        vertical lines, and a list of [x,y] values for diagonal lines."""
        if self.angle%180 == 0:
            return self.getYPos()
        elif self.angle%180 == 90:
            return self.getXPos()
        else:
            return self.getPos()
                
    def setValue(self, v):
        """Set the position of the line. If line is horizontal or vertical, v can be 
        a single value. Otherwise, a 2D coordinate must be specified (list, tuple and 
        QPointF are all acceptable)."""
        self.setPos(v)

    ## broken in 4.7
    #def itemChange(self, change, val):
        #if change in [self.ItemScenePositionHasChanged, self.ItemSceneHasChanged]:
            #self.updateLine()
            #print "update", change
            #print self.getBoundingParents()
        #else:
            #print "ignore", change
        #return GraphicsObject.itemChange(self, change, val)
      
    def boundingRect(self):
        #br = UIGraphicsItem.boundingRect(self)
        br = self.viewRect()
        ## add a 4-pixel radius around the line for mouse interaction.

        px = self.pixelLength(direction=Point(1, 0), ortho=True)  ## get pixel length orthogonal to the line
        if px is None:
            px = 0
        w = (max(4, self.pen.width()/2, self.hoverPen.width()/2)+1) * px
        br.setBottom(-w)
        br.setTop(w)
        return br.normalized()

    def paint(self, p, *args):
        br = self.boundingRect()
        p.setPen(self.currentPen)
        p.drawLine(Point(br.right(), 0), Point(br.left(), 0))
        #p.drawRect(self.boundingRect())
        
    def dataBounds(self, axis, frac=1.0, orthoRange=None):
        if axis == 0:
            return None   ## x axis should never be auto-scaled
        else:
            return (0, 0)
            
    def mouseDragEvent(self, ev):
        if self.movable and ev.button() == QtCore.Qt.LeftButton:
            if ev.isStart():
                self.moving = True
                self.cursorOffset = self.pos()-self.mapToParent(ev.buttonDownPos())
                self.startPosition = self.pos()
                self.currentPen = self.hoverPen
            ev.accept()
            if not self.moving:
                return
            self.setPos(self.cursorOffset + self.mapToParent(ev.pos()))
            self.sigDragged.emit(self)
            if ev.isFinish():
                self.moving = False
                self.sigPositionChangeFinished.emit(self)
                self.currentPen = self.pen

    def mouseClickEvent(self, ev):
        if self.moving and ev.button() == QtCore.Qt.RightButton:
            ev.accept()
            self.setPos(self.startPosition)
            self.moving = False
            self.sigDragged.emit(self)
            self.sigPositionChangeFinished.emit(self)
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

    def hoverEvent(self, ev):
        if (not ev.isExit()) and self.movable and ev.acceptDrags(QtCore.Qt.LeftButton):
            self.setMouseHover(True)
        else:
            self.setMouseHover(False)

    def setMouseHover(self, hover):
        ## Inform the item that the mouse is (not) hovering over it
        if self.mouseHovering == hover:
            return
        self.mouseHovering = hover
        if hover:
            self.currentPen = self.hoverPen
        else:
            self.currentPen = self.pen
        self.update()

    #def hoverEnterEvent(self, ev):
        #print "line hover enter"
        #ev.ignore()
        #self.updateHoverPen()

    #def hoverMoveEvent(self, ev):
        #print "line hover move"
        #ev.ignore()
        #self.updateHoverPen()

    #def hoverLeaveEvent(self, ev):
        #print "line hover leave"
        #ev.ignore()
        #self.updateHoverPen(False)

    #def updateHoverPen(self, hover=None):
        #if hover is None:
            #scene = self.scene()
            #hover = scene.claimEvent(self, QtCore.Qt.LeftButton, scene.Drag)

        #if hover:
            #self.currentPen = fn.mkPen(255, 0,0)
        #else:
            #self.currentPen = self.pen
        #self.update()


class InfiniteLineMenu(QtGui.QMenu):

    def __init__(self, infiniteLine):
        QtGui.QMenu.__init__(self)
        self.infiniteLine = infiniteLine
        if self.infiniteLine.removable:
            remAct = QtGui.QAction("Remove line", self)
            remAct.triggered.connect(self.removeClicked)
            self.addAction(remAct)

        self.editLine = QtGui.QMenu("Edit line")
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
        # === bounds ========
        self.cbBounds = QtGui.QGroupBox("Activate bounds")
        self.cbBounds.setCheckable(True)
        self.lblMin = QtGui.QLabel("min value")
        self.lblMax = QtGui.QLabel("max value")
        self.ledtMin = QtGui.QLineEdit()
        self.ledtMin.setFixedWidth(70)
        tip = "defines the lower bound of the line. If the line is empty\n"
        tip += "or if the input can not be converted into a float, the value\n"
        tip += "of the bound is set to 'None' and the bound is not active."
        self.ledtMin.setToolTip(tip)
        self.ledtMax = QtGui.QLineEdit()
        self.ledtMax.setFixedWidth(70)
        tip = "defines the upper bound of the line. If the line is empty\n"
        tip += "or if the input can not be converted into a float, the value\n"
        tip += "of the bound is set to 'None' and the bound is not active."
        self.ledtMax.setToolTip(tip)
        if self.infiniteLine.useBounds:
            valmin = self.infiniteLine.lowerBound
            valmax = self.infiniteLine.upperBound
            self.cbBounds.setChecked(True)
            if valmin is not None:
                self.ledtMin.setText(str(valmin))
            if valmax is not None:
                self.ledtMax.setText(str(valmax))
        else:
            self.cbBounds.setChecked(False)

        gridLayout = QtGui.QGridLayout()
        gridLayout.addWidget(self.lblMin, 0, 0)
        gridLayout.addWidget(self.ledtMin, 0, 1)
        gridLayout.addWidget(self.lblMax, 1, 0)
        gridLayout.addWidget(self.ledtMax, 1, 1)
        self.cbBounds.setLayout(gridLayout)

        # === label ========
        self.cbText = QtGui.QGroupBox("Visible label")
        self.cbText.setCheckable(True)
        self.cbText.setChecked(False)        
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
        self.sldLocation.setRange(0, 1000)
        self.sldLocation.setValue(int(1000*round(self.infiniteLine.location, 3)))
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
        self.sldShift.setRange(0, 1000)
        self.sldShift.setValue(int(1000*round(self.infiniteLine.shift, 3)))
        self.sldShift.setFixedWidth(70)
        if not self.infiniteLine.activateLocation:
            self.lblLocation.setEnabled(False)
            self.sldLocation.setEnabled(False)
        if not self.infiniteLine.activateShift:
            self.lblShift.setEnabled(False)
            self.sldShift.setEnabled(False)
        if self.infiniteLine.visibleLabel:
            self.cbText.setChecked(True)
            self.sldLocation.setValue(int(1000*self.infiniteLine.location))
            self.sldShift.setValue(int(1000*self.infiniteLine.shift))
        self.lblDigits = QtGui.QLabel("digits")
        self.ledtDigits = QtGui.QLineEdit()
        self.ledtDigits.setValidator(QtGui.QIntValidator())
        self.ledtDigits.setText(str(self.infiniteLine.nDigits))
        self.ledtDigits.setFixedWidth(70)
        tip = "defines the number of digits used to format the label."
        self.ledtDigits.setToolTip(tip)

        gridLayout = QtGui.QGridLayout()
        gridLayout.addWidget(self.lblLocation, 0, 0)
        gridLayout.addWidget(self.sldLocation, 0, 1)
        gridLayout.addWidget(self.lblShift, 1, 0)
        gridLayout.addWidget(self.sldShift, 1, 1)
        gridLayout.addWidget(self.lblDigits, 2, 0)
        gridLayout.addWidget(self.ledtDigits, 2, 1)
        self.cbText.setLayout(gridLayout)

        # === main widget ========
        widget = QtGui.QWidget()
        vBox = QtGui.QVBoxLayout()
        vBox.addWidget(self.cbBounds)
        vBox.addWidget(self.cbText)
        widget.setLayout(vBox)
        widget.setContentsMargins(0, 0, 0, 0)
        widget.layout().setContentsMargins(0, 0, 0, 0)

        # if we only display the line, we hide the label and the specific part 
        # of the contextMenu related to the label
        if self.infiniteLine.onlyLine:
            self.infiniteLine.visibleLabel = False
            QtCore.QTimer.singleShot(0, lambda: self.infiniteLine.sigVisibilityChanged.emit(False))
            self.cbText.hide()            

        return widget

    def updateLocation(self, value):
        floatValue = fn.remapSlider(value, 0, 1000)
        self.infiniteLine.location = floatValue
        self.infiniteLine.sigLocationChanged.emit(floatValue)

    def updateShift(self, value):
        floatValue = fn.remapSlider(value, 0, 1000)
        self.infiniteLine.shift = floatValue
        self.infiniteLine.sigShiftChanged.emit(floatValue)

    def updateStateText(self, state):
        self.infiniteLine.visibleLabel = state
        if state and not self.infiniteLine.activateLocation:
            self.lblLocation.setEnabled(False)
            self.sldLocation.setEnabled(False)
        if state and not self.infiniteLine.activateShift:
            self.lblShift.setEnabled(False)
            self.sldShift.setEnabled(False)
        self.infiniteLine.sigVisibilityChanged.emit(state)

    def updateStateBounds(self, state):
        bound = [None, None]
        if state:
            bound = [self.infiniteLine.lowerBound, self.infiniteLine.upperBound]
        self.infiniteLine.useBounds = state
        self.infiniteLine.setBounds(bound)
        self.infiniteLine.sigBoundsActivated.emit(state)
        self.infiniteLine.sigBoundsChanged.emit(bound)

    def updateMinValue(self, value):
        try:
            minval = float(value)
        except:
            minval = None
        self.infiniteLine.lowerBound = minval
        currentPos = self.infiniteLine.value()
        if minval is not None and currentPos < minval:
            self.infiniteLine.setPos(minval)
        self.updateBounds()

    def updateMaxValue(self, value):
        try:
            maxval = float(value)
        except:
            maxval = None
        self.infiniteLine.upperBound = maxval
        currentPos = self.infiniteLine.value()
        if maxval is not None and currentPos > maxval:
            self.infiniteLine.setPos(maxval)
        self.updateBounds()

    def updateDigits(self):
        digits = self.ledtDigits.text()
        if digits == "":
            return
        else:
            self.infiniteLine.sigDigitsChanged.emit(int(digits))

    def updateBounds(self):
        lowerBound = self.infiniteLine.lowerBound
        upperBound = self.infiniteLine.upperBound
        self.infiniteLine.setBounds([lowerBound, upperBound])
        self.infiniteLine.sigBoundsChanged.emit([lowerBound, upperBound])

    def removeClicked(self):
        ## Send remove event only after we have exited the menu event handler
        QtCore.QTimer.singleShot(0, lambda: self.infiniteLine.sigRemoveRequested.emit(self))
