import warnings
import weakref
from time import perf_counter, perf_counter_ns

from .. import debug as debug
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets, isQObjectAlive
from .mouseEvents import HoverEvent, MouseClickEvent, MouseDragEvent

getMillis = lambda: perf_counter_ns() // 10 ** 6


__all__ = ['GraphicsScene']

class GraphicsScene(QtWidgets.QGraphicsScene):
    """
    Extension of QGraphicsScene that implements a complete, parallel mouse event system.
    (It would have been preferred to just alter the way QGraphicsScene creates and delivers 
    events, but this turned out to be impossible because the constructor for QGraphicsMouseEvent
    is private)
    
      *  Generates MouseClicked events in addition to the usual press/move/release events.
         (This works around a problem where it is impossible to have one item respond to a
         drag if another is watching for a click.)
      *  Adjustable radius around click that will catch objects so you don't have to click *exactly* over small/thin objects
      *  Global context menu--if an item implements a context menu, then its parent(s) may also add items to the menu.
      *  Allows items to decide _before_ a mouse click which item will be the recipient of mouse events.
         This lets us indicate unambiguously to the user which item they are about to click/drag on
      *  Eats mouseMove events that occur too soon after a mouse press.
      *  Reimplements items() and itemAt() to circumvent PyQt bug

    ====================== ====================================================================
    **Signals**
    sigMouseClicked(event) Emitted when the mouse is clicked over the scene. Use ev.pos() to
                           get the click position relative to the item that was clicked on,
                           or ev.scenePos() to get the click position in scene coordinates.
                           See :class:`pyqtgraph.GraphicsScene.mouseEvents.MouseClickEvent`.                        
    sigMouseMoved(pos)     Emitted when the mouse cursor moves over the scene. The position
                           is given in scene coordinates.
    sigMouseHover(items)   Emitted when the mouse is moved over the scene. Items is a list
                           of items under the cursor.
    sigItemAdded(item)     Emitted when an item is added via addItem(). The item is given.
    sigItemRemoved(item)   Emitted when an item is removed via removeItem(). The item is given.
    ====================== ====================================================================
    
    Mouse interaction is as follows:
    
    1) Every time the mouse moves, the scene delivers both the standard hoverEnter/Move/LeaveEvents 
       as well as custom HoverEvents. 
    2) Items are sent HoverEvents in Z-order and each item may optionally call event.acceptClicks(button), 
       acceptDrags(button) or both. If this method call returns True, this informs the item that _if_ 
       the user clicks/drags the specified mouse button, the item is guaranteed to be the 
       recipient of click/drag events (the item may wish to change its appearance to indicate this).
       If the call to acceptClicks/Drags returns False, then the item is guaranteed to *not* receive
       the requested event (because another item has already accepted it). 
    3) If the mouse is clicked, a mousePressEvent is generated as usual. If any items accept this press event, then
       No click/drag events will be generated and mouse interaction proceeds as defined by Qt. This allows
       items to function properly if they are expecting the usual press/move/release sequence of events.
       (It is recommended that items do NOT accept press events, and instead use click/drag events)
       Note: The default implementation of QGraphicsItem.mousePressEvent will *accept* the event if the 
       item is has its Selectable or Movable flags enabled. You may need to override this behavior.
    4) If no item accepts the mousePressEvent, then the scene will begin delivering mouseDrag and/or mouseClick events.
       If the mouse is moved a sufficient distance (or moved slowly enough) before the button is released, 
       then a mouseDragEvent is generated.
       If no drag events are generated before the button is released, then a mouseClickEvent is generated. 
    5) Click/drag events are delivered to the item that called acceptClicks/acceptDrags on the HoverEvent
       in step 1. If no such items exist, then the scene attempts to deliver the events to items near the event. 
       ClickEvents may be delivered in this way even if no
       item originally claimed it could accept the click. DragEvents may only be delivered this way if it is the initial
       move in a drag.
    """
    
    sigMouseHover = QtCore.Signal(object)   ## emits a list of objects hovered over
    sigMouseMoved = QtCore.Signal(object)   ## emits position of mouse on every move
    sigMouseClicked = QtCore.Signal(object)   ## emitted when mouse is clicked. Check for event.isAccepted() to see whether the event has already been acted on.
    
    sigPrepareForPaint = QtCore.Signal()  ## emitted immediately before the scene is about to be rendered

    sigItemAdded = QtCore.Signal(object)  ## emits the item object just added
    sigItemRemoved = QtCore.Signal(object)  ## emits the item object just removed

    _addressCache = weakref.WeakValueDictionary()
    
    ExportDirectory = None

    def __init__(self, clickRadius: int = 2, moveDistance=5, parent=None):
        QtWidgets.QGraphicsScene.__init__(self, parent)
        self.setClickRadius(clickRadius)
        self.setMoveDistance(moveDistance)
        self.exportDirectory = None
        
        self.clickEvents = []
        self.dragButtons = []
        self.mouseGrabber = None
        self.dragItem = None
        self.lastDrag = None
        self.hoverItems = weakref.WeakKeyDictionary()
        self.lastHoverEvent = None
        self.minDragTime = 0.5  # drags shorter than 0.5 sec are interpreted as clicks
        
        self.contextMenu = [QtGui.QAction(QtCore.QCoreApplication.translate("GraphicsScene", "Export..."), self)]
        self.contextMenu[0].triggered.connect(self.showExportDialog)
        
        self.exportDialog = None
        self._lastMoveEventTime = 0
        
    def render(self, *args):
        self.prepareForPaint()
        return QtWidgets.QGraphicsScene.render(self, *args)

    @QtCore.Slot()
    def prepareForPaint(self):
        """Called before every render. This method will inform items that the scene is about to
        be rendered by emitting sigPrepareForPaint.
        
        This allows items to delay expensive processing until they know a paint will be required."""
        self.sigPrepareForPaint.emit()
    

    def setClickRadius(self, r: int):
        """
        Set the distance away from mouse clicks to search for interacting items.
        When clicking, the scene searches first for items that directly intersect the click position
        followed by any other items that are within a rectangle that extends r pixels away from the 
        click position. 
        """
        self._clickRadius = int(r)
        
    def setMoveDistance(self, d):
        """
        Set the distance the mouse must move after a press before mouseMoveEvents will be delivered.
        This ensures that clicks with a small amount of movement are recognized as clicks instead of
        drags.
        """
        self._moveDistance = d

    def mousePressEvent(self, ev):
        super().mousePressEvent(ev)
        if self.mouseGrabberItem() is None:  ## nobody claimed press; we are free to generate drag/click events
            if self.lastHoverEvent is not None:
                # If the mouse has moved since the last hover event, send a new one.
                # This can happen if a context menu is open while the mouse is moving.
                if ev.scenePos() != self.lastHoverEvent.scenePos():
                    self.sendHoverEvents(ev)
            
            self.clickEvents.append(MouseClickEvent(ev))
            
            ## set focus on the topmost focusable item under this click
            items = self.items(ev.scenePos())
            for i in items:
                if i.isEnabled() and i.isVisible() and (i.flags() & i.GraphicsItemFlag.ItemIsFocusable):
                    i.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
                    break

    def _moveEventIsAllowed(self):
        # For ignoring events that are too close together

        # Max number of events per second
        rateLimit = getConfigOption('mouseRateLimit')
        if rateLimit <= 0:
            return True

        # Delay between events (in milliseconds)
        delay = 1000.0 / rateLimit
        if getMillis() - self._lastMoveEventTime >= delay:
            return True
        return False

        
    def mouseMoveEvent(self, ev):
        # ignore high frequency events
        if self._moveEventIsAllowed():
            self._lastMoveEventTime = getMillis()
            self.sigMouseMoved.emit(ev.scenePos())

            # First allow QGraphicsScene to eliver hoverEvent/Move/Exit Events
            super().mouseMoveEvent(ev)
            # Next Deliver our own Hover Events
            self.sendHoverEvents(ev)
            if ev.buttons():
                # button is pressed' send mouseDragEvents
                if self.mouseGrabberItem() is None:
                    now = perf_counter()
                    init = False
                    ## keep track of which buttons are involved in dragging
                    for btn in [QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.MiddleButton, QtCore.Qt.MouseButton.RightButton]:
                        if not (ev.buttons() & btn):
                            continue
                        if btn not in self.dragButtons:  ## see if we've dragged far enough yet
                            cev = [e for e in self.clickEvents if e.button() == btn]
                            if cev:
                                cev = cev[0]
                                dist = Point(ev.scenePos() - cev.scenePos()).length()
                                if dist == 0 or (dist < self._moveDistance and now - cev.time() < self.minDragTime):
                                    continue
                                init = init or (len(self.dragButtons) == 0)  ## If this is the first button to be dragged, then init=True
                                self.dragButtons.append(btn)
                    ## if we have dragged buttons, deliver a drag event
                    if len(self.dragButtons) > 0:
                        if self.sendDragEvent(ev, init=init):
                            ev.accept()

        else:
            super().mouseMoveEvent(ev)
            # if you do not accept event (which is ignored) then cursor will disappear
            ev.accept()
                
    def leaveEvent(self, ev):  ## inform items that mouse is gone
        if len(self.dragButtons) == 0:
            self.sendHoverEvents(ev, exitOnly=True)
                
    def mouseReleaseEvent(self, ev):
        if self.mouseGrabberItem() is None:
            if ev.button() in self.dragButtons:
                if self.sendDragEvent(ev, final=True):
                    #print "sent drag event"
                    ev.accept()
                self.dragButtons.remove(ev.button())
            else:
                cev = [e for e in self.clickEvents if e.button() == ev.button()]
                if cev:
                    if self.sendClickEvent(cev[0]):
                        ev.accept()
                    try:
                        self.clickEvents.remove(cev[0])
                    except ValueError:
                        warnings.warn(
                            ("A ValueError can occur here with errant "
                             "QApplication.processEvent() calls, see "
                            "https://github.com/pyqtgraph/pyqtgraph/pull/2580 "
                            "for more information."),
                            RuntimeWarning,
                            stacklevel=2
                        )
        if not ev.buttons():
            self.dragItem = None
            self.dragButtons = []
            self.clickEvents = []
            self.lastDrag = None
        super().mouseReleaseEvent(ev)
        
        self.sendHoverEvents(ev)  ## let items prepare for next click/drag

    def mouseDoubleClickEvent(self, ev):
        super().mouseDoubleClickEvent(ev)
        if self.mouseGrabberItem() is None:  ## nobody claimed press; we are free to generate drag/click events
            self.clickEvents.append(MouseClickEvent(ev, double=True))
        
    def sendHoverEvents(self, ev, exitOnly=False):
        ## if exitOnly, then just inform all previously hovered items that the mouse has left.
        
        if exitOnly:
            acceptable=False
            items = []
            event = HoverEvent(None, acceptable)
        else:
            acceptable = not ev.buttons()  ## if we are in mid-drag, do not allow items to accept the hover event.
            event = HoverEvent(ev, acceptable)
            items = self.itemsNearEvent(event, hoverable=True)
            self.sigMouseHover.emit(items)
            
        prevItems = list(self.hoverItems.keys())
            
        for item in items:
            if hasattr(item, 'hoverEvent'):
                event.currentItem = item
                if item not in self.hoverItems:
                    self.hoverItems[item] = None
                    event.enter = True
                else:
                    prevItems.remove(item)
                    event.enter = False
                    
                try:
                    item.hoverEvent(event)
                except:
                    debug.printExc("Error sending hover event:")
        
        event.enter = False
        event.exit = True
        #print "hover exit items:", prevItems
        for item in prevItems:
            event.currentItem = item
            try:
                # NOTE: isQObjectAlive(item) was added for PySide6 where
                #       verlet_chain_demo.py triggers a RuntimeError.
                if isQObjectAlive(item) and item.scene() is self:
                    item.hoverEvent(event)
            except:
                debug.printExc("Error sending hover exit event:")
            finally:
                del self.hoverItems[item]
        
        # Update last hover event unless:
        #   - mouse is dragging (move+buttons); in this case we want the dragged
        #     item to continue receiving events until the drag is over
        #   - event is not a mouse event (QEvent.Type.Leave sometimes appears here)
        if (ev.type() == ev.Type.GraphicsSceneMousePress or 
            (ev.type() == ev.Type.GraphicsSceneMouseMove and not ev.buttons())):
            self.lastHoverEvent = event  ## save this so we can ask about accepted events later.

    def sendDragEvent(self, ev, init=False, final=False):
        ## Send a MouseDragEvent to the current dragItem or to 
        ## items near the beginning of the drag
        event = MouseDragEvent(ev, self.clickEvents[0], self.lastDrag, start=init, finish=final)
        #print "dragEvent: init=", init, 'final=', final, 'self.dragItem=', self.dragItem
        if init and self.dragItem is None:
            if self.lastHoverEvent is not None:
                acceptedItem = self.lastHoverEvent.dragItems().get(event.button(), None)
            else:
                acceptedItem = None
                
            if acceptedItem is not None and acceptedItem.scene() is self:
                #print "Drag -> pre-selected item:", acceptedItem
                self.dragItem = acceptedItem
                event.currentItem = self.dragItem
                try:
                    self.dragItem.mouseDragEvent(event)
                except:
                    debug.printExc("Error sending drag event:")
                    
            else:
                #print "drag -> new item"
                for item in self.itemsNearEvent(event):
                    #print "check item:", item
                    if not item.isVisible() or not item.isEnabled():
                        continue
                    if hasattr(item, 'mouseDragEvent'):
                        event.currentItem = item
                        try:
                            item.mouseDragEvent(event)
                        except:
                            debug.printExc("Error sending drag event:")
                        if event.isAccepted():
                            #print "   --> accepted"
                            self.dragItem = item
                            if item.flags() & item.GraphicsItemFlag.ItemIsFocusable:
                                item.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
                            break
        elif self.dragItem is not None:
            event.currentItem = self.dragItem
            try:
                self.dragItem.mouseDragEvent(event)
            except:
                debug.printExc("Error sending hover exit event:")
            
        self.lastDrag = event
        
        return event.isAccepted()
            
        
    def sendClickEvent(self, ev):
        ## if we are in mid-drag, click events may only go to the dragged item.
        if self.dragItem is not None and hasattr(self.dragItem, 'mouseClickEvent'):
            ev.currentItem = self.dragItem
            self.dragItem.mouseClickEvent(ev)
            
        ## otherwise, search near the cursor
        else:
            if self.lastHoverEvent is not None:
                acceptedItem = self.lastHoverEvent.clickItems().get(ev.button(), None)
            else:
                acceptedItem = None
            if acceptedItem is not None:
                ev.currentItem = acceptedItem
                try:
                    acceptedItem.mouseClickEvent(ev)
                except:
                    debug.printExc("Error sending click event:")
            else:
                for item in self.itemsNearEvent(ev):
                    if not item.isVisible() or not item.isEnabled():
                        continue
                    if hasattr(item, 'mouseClickEvent'):
                        ev.currentItem = item
                        try:
                            item.mouseClickEvent(ev)
                        except:
                            debug.printExc("Error sending click event:")
                            
                        if ev.isAccepted():
                            if item.flags() & item.GraphicsItemFlag.ItemIsFocusable:
                                item.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
                            break
        self.sigMouseClicked.emit(ev)
        return ev.isAccepted()

    def addItem(self, item):
        # extend QGraphicsScene.addItem to emit a sigItemAdded signal
        ret = QtWidgets.QGraphicsScene.addItem(self, item)
        self.sigItemAdded.emit(item)
        return ret

    def removeItem(self, item):
        # extend QGraphicsScene.removeItem to emit a sigItemRemoved signal
        ret = QtWidgets.QGraphicsScene.removeItem(self, item)
        self.sigItemRemoved.emit(item)
        return ret

    def itemsNearEvent(
        self,
        event,
        selMode=QtCore.Qt.ItemSelectionMode.IntersectsItemShape,
        sortOrder=QtCore.Qt.SortOrder.DescendingOrder,
        hoverable=False,
    ):
        """
        Return an iterator that iterates first through the items that directly intersect point (in Z order)
        followed by any other items that are within the scene's click radius.
        """
        view = self.views()[0]
        tr = view.viewportTransform()

        if hasattr(event, "buttonDownScenePos"):
            point = event.buttonDownScenePos()
        else:
            point = event.scenePos()

        ## Sort by descending Z-order (don't trust scene.itms() to do this either)
        ## use 'absolute' z value, which is the sum of all item/parent ZValues
        def absZValue(item):
            if item is None:
                return 0
            return item.zValue() + absZValue(item.parentItem())

        ## Get items, which directly are at the given point (sorted by z-value)
        items_at_point = self.items(point, selMode, sortOrder, tr)
        items_at_point.sort(key=absZValue, reverse=True)

        ## Get items, which are within the click radius around the given point (sorted by z-value)
        r = self._clickRadius
        items_within_radius = []
        rgn = None
        if r > 0:
            rect = view.mapToScene(QtCore.QRect(0, 0, 2 * r, 2 * r)).boundingRect()
            w = rect.width()
            h = rect.height()
            rgn = QtCore.QRectF(point.x() - w / 2, point.y() - h / 2, w, h)
            items_within_radius = self.items(rgn, selMode, sortOrder, tr)
            items_within_radius.sort(key=absZValue, reverse=True)
            # Remove items, which are already in the other list
            for item in items_at_point:
                if item in items_within_radius:
                    items_within_radius.remove(item)

        ## Put both groups of items together, but in the correct order
        ## The items directly at the given point shall have higher priority
        all_items = items_at_point + items_within_radius

        ## Remove items, which we don't want, due to several reasons
        selected_items = []
        for item in all_items:
            if hoverable and not hasattr(item, "hoverEvent"):
                continue
            if item.scene() is not self:
                continue
            shape = item.shape()  # Note: default shape() returns boundingRect()
            if shape is None:
                continue
            # Remove items whose shape does not contain point or region
            # (scene.items() apparently sucks at this)
            if (
                rgn is not None
                and shape.intersects(item.mapFromScene(rgn).boundingRect())
            ) or shape.contains(item.mapFromScene(point)):
                selected_items.append(item)

        return selected_items

    def getViewWidget(self):
        return self.views()[0]
    
    def addParentContextMenus(self, item, menu, event):
        """
        Can be called by any item in the scene to expand its context menu to include parent context menus.
        Parents may implement getContextMenus to add new menus / actions to the existing menu.
        getContextMenus must accept 1 argument (the event that generated the original menu) and
        return a single QMenu or a list of QMenus.
        
        The final menu will look like:
        
            |    Original Item 1
            |    Original Item 2
            |    ...
            |    Original Item N
            |    ------------------
            |    Parent Item 1
            |    Parent Item 2
            |    ...
            |    Grandparent Item 1
            |    ...
            
        
        ==============  ==================================================
        **Arguments:**
        item            The item that initially created the context menu 
                        (This is probably the item making the call to this function)
        menu            The context menu being shown by the item
        event           The original event that triggered the menu to appear.
        ==============  ==================================================
        """

        menusToAdd = []
        while item is not self:
            item = item.parentItem()
            if item is None:
                item = self
            if not hasattr(item, "getContextMenus"):
                continue
            subMenus = item.getContextMenus(event) or []
            if isinstance(subMenus, list): ## so that some items (like FlowchartViewBox) can return multiple menus
                menusToAdd.extend(subMenus)
            else:
                menusToAdd.append(subMenus)
        # Filter out options that were previously added
        existingActions = menu.actions()
        actsToAdd = []
        for menuOrAct in menusToAdd:
            if isinstance(menuOrAct, QtWidgets.QMenu):
                menuOrAct = menuOrAct.menuAction()
            elif not isinstance(menuOrAct, QtGui.QAction):
                raise Exception(
                    f"Cannot add object {menuOrAct} (type={type(menuOrAct)}) to QMenu."
                )
            if menuOrAct not in existingActions:
                actsToAdd.append(menuOrAct)

        if actsToAdd:
            menu.addSeparator()

        menu.addActions(actsToAdd)
        return menu

    def getContextMenus(self, event):
        self.contextMenuItem = event.acceptedItem
        return self.contextMenu

    @QtCore.Slot()
    def showExportDialog(self):
        if self.exportDialog is None:
            from . import exportDialog
            self.exportDialog = exportDialog.ExportDialog(self)
        self.exportDialog.show(self.contextMenuItem)
