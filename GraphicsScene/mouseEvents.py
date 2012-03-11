from pyqtgraph.Point import Point
from pyqtgraph.Qt import QtCore, QtGui
import weakref
import pyqtgraph.ptime as ptime

class MouseDragEvent:
    def __init__(self, moveEvent, pressEvent, lastEvent, start=False, finish=False):
        self.start = start
        self.finish = finish
        self.accepted = False
        self.currentItem = None
        self._buttonDownScenePos = {}
        self._buttonDownScreenPos = {}
        for btn in [QtCore.Qt.LeftButton, QtCore.Qt.MidButton, QtCore.Qt.RightButton]:
            self._buttonDownScenePos[int(btn)] = moveEvent.buttonDownScenePos(btn)
            self._buttonDownScreenPos[int(btn)] = moveEvent.buttonDownScreenPos(btn)
        self._scenePos = moveEvent.scenePos()
        self._screenPos = moveEvent.screenPos()
        if lastEvent is None:
            self._lastScenePos = pressEvent.scenePos()
            self._lastScreenPos = pressEvent.screenPos()
        else:
            self._lastScenePos = lastEvent.scenePos()
            self._lastScreenPos = lastEvent.screenPos()
        self._buttons = moveEvent.buttons()
        self._button = pressEvent.button()
        self._modifiers = moveEvent.modifiers()
        
    def accept(self):
        self.accepted = True
        self.acceptedItem = self.currentItem
        
    def ignore(self):
        self.accepted = False
    
    def isAccepted(self):
        return self.accepted
    
    def scenePos(self):
        return Point(self._scenePos)
    
    def screenPos(self):
        return Point(self._screenPos)
    
    def buttonDownScenePos(self, btn=None):
        if btn is None:
            btn = self.button()
        return Point(self._buttonDownScenePos[int(btn)])
    
    def buttonDownScreenPos(self, btn=None):
        if btn is None:
            btn = self.button()
        return Point(self._buttonDownScreenPos[int(btn)])
    
    def lastScenePos(self):
        return Point(self._lastScenePos)
    
    def lastScreenPos(self):
        return Point(self._lastScreenPos)
    
    def buttons(self):
        return self._buttons
        
    def button(self):
        """Return the button that initiated the drag (may be different from the buttons currently pressed)"""
        return self._button
        
    def pos(self):
        return Point(self.currentItem.mapFromScene(self._scenePos))
    
    def lastPos(self):
        return Point(self.currentItem.mapFromScene(self._lastScenePos))
        
    def buttonDownPos(self, btn=None):
        if btn is None:
            btn = self.button()
        return Point(self.currentItem.mapFromScene(self._buttonDownScenePos[int(btn)]))
    
    def isStart(self):
        return self.start
        
    def isFinish(self):
        return self.finish

    def __repr__(self):
        lp = self.lastPos()
        p = self.pos()
        return "<MouseDragEvent (%g,%g)->(%g,%g) buttons=%d start=%s finish=%s>" % (lp.x(), lp.y(), p.x(), p.y(), int(self.buttons()), str(self.isStart()), str(self.isFinish()))
        
    def modifiers(self):
        return self._modifiers



class MouseClickEvent:
    def __init__(self, pressEvent, double=False):
        self.accepted = False
        self.currentItem = None
        self._double = double
        self._scenePos = pressEvent.scenePos()
        self._screenPos = pressEvent.screenPos()
        self._button = pressEvent.button()
        self._buttons = pressEvent.buttons()
        self._modifiers = pressEvent.modifiers()
        self._time = ptime.time()
        
        
    def accept(self):
        self.accepted = True
        self.acceptedItem = self.currentItem
        
    def ignore(self):
        self.accepted = False
    
    def isAccepted(self):
        return self.accepted
    
    def scenePos(self):
        return Point(self._scenePos)
    
    def screenPos(self):
        return Point(self._screenPos)
    
    def buttons(self):
        return self._buttons
    
    def button(self):
        return self._button
    
    def double(self):
        return self._double

    def pos(self):
        return Point(self.currentItem.mapFromScene(self._scenePos))
    
    def lastPos(self):
        return Point(self.currentItem.mapFromScene(self._lastScenePos))
        
    def modifiers(self):
        return self._modifiers

    def __repr__(self):
        p = self.pos()
        return "<MouseClickEvent (%g,%g) button=%d>" % (p.x(), p.y(), int(self.button()))
        
    def time(self):
        return self._time



class HoverEvent:
    """
    This event class both informs items that the mouse cursor is nearby and allows items to 
    communicate with one another about whether each item will accept _potential_ mouse events. 
    
    It is common for multiple overlapping items to receive hover events and respond by changing 
    their appearance. This can be misleading to the user since, in general, only one item will
    respond to mouse events. To avoid this, items make calls to event.acceptClicks(button) 
    and/or acceptDrags(button).
    
    Each item may make multiple calls to acceptClicks/Drags, each time for a different button. 
    If the method returns True, then the item is guaranteed to be
    the recipient of the claimed event IF the user presses the specified mouse button before
    moving. If claimEvent returns False, then this item is guaranteed NOT to get the specified
    event (because another has already claimed it) and the item should change its appearance 
    accordingly.
    
    event.isEnter() returns True if the mouse has just entered the item's shape;
    event.isExit() returns True if the mouse has just left.
    """
    def __init__(self, moveEvent, acceptable):
        self.enter = False
        self.acceptable = acceptable
        self.exit = False
        self.__clickItems = weakref.WeakValueDictionary()
        self.__dragItems = weakref.WeakValueDictionary()
        self.currentItem = None
        if moveEvent is not None:
            self._scenePos = moveEvent.scenePos()
            self._screenPos = moveEvent.screenPos()
            self._lastScenePos = moveEvent.lastScenePos()
            self._lastScreenPos = moveEvent.lastScreenPos()
            self._buttons = moveEvent.buttons()
            self._modifiers = moveEvent.modifiers()
        else:
            self.exit = True
            
        
        
    def isEnter(self):
        return self.enter
        
    def isExit(self):
        return self.exit
        
    def acceptClicks(self, button):
        """"""
        if not self.acceptable:
            return False
        if button not in self.__clickItems:
            self.__clickItems[button] = self.currentItem
            return True
        return False
        
    def acceptDrags(self, button):
        if not self.acceptable:
            return False
        if button not in self.__dragItems:
            self.__dragItems[button] = self.currentItem
            return True
        return False
        
    def scenePos(self):
        return Point(self._scenePos)
    
    def screenPos(self):
        return Point(self._screenPos)
    
    def lastScenePos(self):
        return Point(self._lastScenePos)
    
    def lastScreenPos(self):
        return Point(self._lastScreenPos)
    
    def buttons(self):
        return self._buttons
        
    def pos(self):
        return Point(self.currentItem.mapFromScene(self._scenePos))
    
    def lastPos(self):
        return Point(self.currentItem.mapFromScene(self._lastScenePos))

    def __repr__(self):
        lp = self.lastPos()
        p = self.pos()
        return "<HoverEvent (%g,%g)->(%g,%g) buttons=%d enter=%s exit=%s>" % (lp.x(), lp.y(), p.x(), p.y(), int(self.buttons()), str(self.isEnter()), str(self.isExit()))
        
    def modifiers(self):
        return self._modifiers
    
    def clickItems(self):
        return self.__clickItems
        
    def dragItems(self):
        return self.__dragItems
        
    
    