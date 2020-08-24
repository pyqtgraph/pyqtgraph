# -*- coding: utf-8 -*-
import weakref
from ..Qt import QtCore, QtGui
from .Container import *
from .DockDrop import *
from .Dock import Dock
from .. import debug as debug
from ..python2_3 import basestring


class DockArea(Container, QtGui.QWidget, DockDrop):
    def __init__(self, parent=None, temporary=False, home=None):
        Container.__init__(self, self)
        QtGui.QWidget.__init__(self, parent=parent)
        DockDrop.__init__(self, allowedAreas=['left', 'right', 'top', 'bottom'])
        self.layout = QtGui.QVBoxLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.setSpacing(0)
        self.setLayout(self.layout)
        self.docks = weakref.WeakValueDictionary()
        self.topContainer = None
        self.raiseOverlay()
        self.temporary = temporary
        self.tempAreas = []
        self.home = home
        
    def type(self):
        return "top"
        
    def addDock(self, dock=None, position='bottom', relativeTo=None, **kwds):
        """Adds a dock to this area.
        
        ============== =================================================================
        **Arguments:**
        dock           The new Dock object to add. If None, then a new Dock will be 
                       created.
        position       'bottom', 'top', 'left', 'right', 'above', or 'below'
        relativeTo     If relativeTo is None, then the new Dock is added to fill an 
                       entire edge of the window. If relativeTo is another Dock, then 
                       the new Dock is placed adjacent to it (or in a tabbed 
                       configuration for 'above' and 'below'). 
        ============== =================================================================
        
        All extra keyword arguments are passed to Dock.__init__() if *dock* is
        None.        
        """
        if dock is None:
            dock = Dock(**kwds)
            
        # store original area that the dock will return to when un-floated
        if not self.temporary:
            dock.orig_area = self
        
        
        ## Determine the container to insert this dock into.
        ## If there is no neighbor, then the container is the top.
        if relativeTo is None or relativeTo is self:
            if self.topContainer is None:
                container = self
                neighbor = None
            else:
                container = self.topContainer
                neighbor = None
        else:
            if isinstance(relativeTo, basestring):
                relativeTo = self.docks[relativeTo]
            container = self.getContainer(relativeTo)
            if container is None:
                raise TypeError("Dock %s is not contained in a DockArea; cannot add another dock relative to it." % relativeTo)
            neighbor = relativeTo
        
        ## what container type do we need?
        neededContainer = {
            'bottom': 'vertical',
            'top': 'vertical',
            'left': 'horizontal',
            'right': 'horizontal',
            'above': 'tab',
            'below': 'tab'
        }[position]
        
        ## Can't insert new containers into a tab container; insert outside instead.
        if neededContainer != container.type() and container.type() == 'tab':
            neighbor = container
            container = container.container()
            
        ## Decide if the container we have is suitable.
        ## If not, insert a new container inside.
        if neededContainer != container.type():
            if neighbor is None:
                container = self.addContainer(neededContainer, self.topContainer)
            else:
                container = self.addContainer(neededContainer, neighbor)
            
        ## Insert the new dock before/after its neighbor
        insertPos = {
            'bottom': 'after',
            'top': 'before',
            'left': 'before',
            'right': 'after',
            'above': 'before',
            'below': 'after'
        }[position]
        #print "request insert", dock, insertPos, neighbor
        old = dock.container()
        container.insert(dock, insertPos, neighbor)
        self.docks[dock.name()] = dock
        if old is not None:
            old.apoptose()
        
        return dock
        
    def moveDock(self, dock, position, neighbor):
        """
        Move an existing Dock to a new location. 
        """
        ## Moving to the edge of a tabbed dock causes a drop outside the tab box
        if position in ['left', 'right', 'top', 'bottom'] and neighbor is not None and neighbor.container() is not None and neighbor.container().type() == 'tab':
            neighbor = neighbor.container()
        self.addDock(dock, position, neighbor)
        
    def getContainer(self, obj):
        if obj is None:
            return self
        return obj.container()
        
    def makeContainer(self, typ):
        if typ == 'vertical':
            new = VContainer(self)
        elif typ == 'horizontal':
            new = HContainer(self)
        elif typ == 'tab':
            new = TContainer(self)
        return new
        
    def addContainer(self, typ, obj):
        """Add a new container around obj"""
        new = self.makeContainer(typ)
        
        container = self.getContainer(obj)
        container.insert(new, 'before', obj)
        #print "Add container:", new, " -> ", container
        if obj is not None:
            new.insert(obj)
        self.raiseOverlay()
        return new
    
    def insert(self, new, pos=None, neighbor=None):
        if self.topContainer is not None:
            # Adding new top-level container; addContainer() should
            # take care of giving the old top container a new home.
            self.topContainer.containerChanged(None)
        self.layout.addWidget(new)
        new.containerChanged(self)
        self.topContainer = new
        self.raiseOverlay()
        
    def count(self):
        if self.topContainer is None:
            return 0
        return 1
        
    def resizeEvent(self, ev):
        self.resizeOverlay(self.size())
        
    def addTempArea(self):
        if self.home is None:
            area = DockArea(temporary=True, home=self)
            self.tempAreas.append(area)
            win = TempAreaWindow(area)
            area.win = win
            win.show()
        else:
            area = self.home.addTempArea()
        #print "added temp area", area, area.window()
        return area
        
    def floatDock(self, dock):
        """Removes *dock* from this DockArea and places it in a new window."""
        area = self.addTempArea()
        area.win.resize(dock.size())
        area.moveDock(dock, 'top', None)
        
    def removeTempArea(self, area):
        self.tempAreas.remove(area)
        #print "close window", area.window()
        area.window().close()
        
    def saveState(self):
        """
        Return a serialized (storable) representation of the state of
        all Docks in this DockArea."""

        if self.topContainer is None:
            main = None
        else:
            main = self.childState(self.topContainer)

        state = {'main': main, 'float': []}
        for a in self.tempAreas:
            geo = a.win.geometry()
            geo = (geo.x(), geo.y(), geo.width(), geo.height())
            state['float'].append((a.saveState(), geo))
        return state
        
    def childState(self, obj):
        if isinstance(obj, Dock):
            return ('dock', obj.name(), {})
        else:
            childs = []
            for i in range(obj.count()):
                childs.append(self.childState(obj.widget(i)))
            return (obj.type(), childs, obj.saveState())
        
    def restoreState(self, state, missing='error', extra='bottom'):
        """
        Restore Dock configuration as generated by saveState.
        
        This function does not create any Docks--it will only 
        restore the arrangement of an existing set of Docks.
        
        By default, docks that are described in *state* but do not exist
        in the dock area will cause an exception to be raised. This behavior
        can be changed by setting *missing* to 'ignore' or 'create'.
        
        Extra docks that are in the dockarea but that are not mentioned in
        *state* will be added to the bottom of the dockarea, unless otherwise
        specified by the *extra* argument.
        """
        
        ## 1) make dict of all docks and list of existing containers
        containers, docks = self.findAll()
        oldTemps = self.tempAreas[:]
        #print "found docks:", docks
        
        ## 2) create container structure, move docks into new containers
        if state['main'] is not None:
            self.buildFromState(state['main'], docks, self, missing=missing)
        
        ## 3) create floating areas, populate
        for s in state['float']:
            a = self.addTempArea()
            a.buildFromState(s[0]['main'], docks, a, missing=missing)
            a.win.setGeometry(*s[1])
            a.apoptose()  # ask temp area to close itself if it is empty
        
        ## 4) Add any remaining docks to a float
        for d in docks.values():
            if extra == 'float':
                a = self.addTempArea()
                a.addDock(d, 'below')
            else:
                self.moveDock(d, extra, None)
        
        #print "\nKill old containers:"
        ## 5) kill old containers
        for c in containers:
            c.close()
        for a in oldTemps:
            a.apoptose()

    def buildFromState(self, state, docks, root, depth=0, missing='error'):
        typ, contents, state = state
        pfx = "  " * depth
        if typ == 'dock':
            try:
                obj = docks[contents]
                del docks[contents]
            except KeyError:
                if missing == 'error':
                    raise Exception('Cannot restore dock state; no dock with name "%s"' % contents)
                elif missing == 'create':
                    obj = Dock(name=contents)
                elif missing == 'ignore':
                    return
                else:
                    raise ValueError('"missing" argument must be one of "error", "create", or "ignore".')

        else:
            obj = self.makeContainer(typ)
            
        root.insert(obj, 'after')
        #print pfx+"Add:", obj, " -> ", root
        
        if typ != 'dock':
            for o in contents:
                self.buildFromState(o, docks, obj, depth+1, missing=missing)
            # remove this container if possible. (there are valid situations when a restore will
            # generate empty containers, such as when using missing='ignore')
            obj.apoptose(propagate=False)
            obj.restoreState(state)  ## this has to be done later?     

    def findAll(self, obj=None, c=None, d=None):
        if obj is None:
            obj = self.topContainer
        
        ## check all temp areas first
        if c is None:
            c = []
            d = {}
            for a in self.tempAreas:
                c1, d1 = a.findAll()
                c.extend(c1)
                d.update(d1)
        
        if isinstance(obj, Dock):
            d[obj.name()] = obj
        elif obj is not None:
            c.append(obj)
            for i in range(obj.count()):
                o2 = obj.widget(i)
                c2, d2 = self.findAll(o2)
                c.extend(c2)
                d.update(d2)
        return (c, d)

    def apoptose(self, propagate=True):
        # remove top container if possible, close this area if it is temporary.
        #print "apoptose area:", self.temporary, self.topContainer, self.topContainer.count()
        if self.topContainer is None or self.topContainer.count() == 0:
            self.topContainer = None
            if self.temporary and self.home:
                self.home.removeTempArea(self)
                #self.close()
                
    def clear(self):
        docks = self.findAll()[1]
        for dock in docks.values():
            dock.close()
            
    ## PySide bug: We need to explicitly redefine these methods
    ## or else drag/drop events will not be delivered.
    def dragEnterEvent(self, *args):
        DockDrop.dragEnterEvent(self, *args)

    def dragMoveEvent(self, *args):
        DockDrop.dragMoveEvent(self, *args)

    def dragLeaveEvent(self, *args):
        DockDrop.dragLeaveEvent(self, *args)

    def dropEvent(self, *args):
        DockDrop.dropEvent(self, *args)

    def printState(self, state=None, name='Main'):
        # for debugging
        if state is None:
            state = self.saveState()
        print("=== %s dock area ===" % name)
        if state['main'] is None:
            print("   (empty)")
        else:
            self._printAreaState(state['main'])
        for i, float in enumerate(state['float']):
            self.printState(float[0], name='float %d' % i)

    def _printAreaState(self, area, indent=0):
        if area[0] == 'dock':
            print("  " * indent + area[0] + " " + str(area[1:]))
            return
        else:
            print("  " * indent + area[0])
            for ch in area[1]:
                self._printAreaState(ch, indent+1)



class TempAreaWindow(QtGui.QWidget):
    def __init__(self, area, **kwargs):
        QtGui.QWidget.__init__(self, **kwargs)
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.dockarea = area
        self.layout.addWidget(area)

    def closeEvent(self, *args):
        # restore docks to their original area
        docks = self.dockarea.findAll()[1]
        for dock in docks.values():
            if hasattr(dock, 'orig_area'):
                dock.orig_area.addDock(dock, )
        # clear dock area, and close remaining docks
        self.dockarea.clear()
        QtGui.QWidget.closeEvent(self, *args)
