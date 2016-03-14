from ..Qt import QtGui, QtCore, isQObjectAlive
from ..GraphicsScene import GraphicsScene
from ..Point import Point
from .. import functions as fn
import weakref
import operator
from ..util.lru_cache import LRUCache


class GraphicsItem(object):
    """
    **Bases:** :class:`object`

    Abstract class providing useful methods to GraphicsObject and GraphicsWidget.
    (This is required because we cannot have multiple inheritance with QObject subclasses.)

    A note about Qt's GraphicsView framework:

    The GraphicsView system places a lot of emphasis on the notion that the graphics within
    the scene should be device independent--you should be able to take the same graphics
    and display them on screens of different resolutions, printers, export to SVG, etc.
    This is nice in principle, but causes me a lot of headache in practice.
    It means that I have to circumvent all the device-independent expectations any time I
    want to operate in pixel coordinates rather than arbitrary scene coordinates.
    A lot of the code in GraphicsItem is devoted to this task--keeping track of view widgets
    and device transforms, computing the size and shape of a pixel in local item coordinates,
    etc.
    Note that in item coordinates, a pixel does not have to be square or even rectangular,
    so just asking how to increase a bounding rect by 2px can be a rather complex task.
    """
    _pixelVectorGlobalCache = LRUCache(100, 70)

    def __init__(self, register=True):
        if not hasattr(self, '_qtBaseClass'):
            for b in self.__class__.__bases__:
                if issubclass(b, QtGui.QGraphicsItem):
                    self.__class__._qtBaseClass = b
                    break
        if not hasattr(self, '_qtBaseClass'):
            raise Exception('Could not determine Qt base class for GraphicsItem: %s' % str(self))

        self._pixelVectorCache = [None, None]
        self._viewBox = None
        self._connectedView = None
        self._exportOpts = False   ## If False, not currently exporting. Otherwise, contains dict of export options.
        #if register:
        #    GraphicsScene.registerObject(self)  ## workaround for pyqt bug in graphicsscene.items()

    '''
    def getViewBox(self):
        """
        Return the first ViewBox or GraphicsView which bounds this item's visible space.
        If this item is not contained within a ViewBox, then the GraphicsView is returned.
        If the item is contained inside nested ViewBoxes, then the inner-most ViewBox is returned.
        The result is cached; clear the cache with forgetViewBox()
        """
        if self._viewBox is None:
            p = self
            while True:
                try:
                    p = p.parentItem()
                except RuntimeError:  ## sometimes happens as items are being removed from a scene and collected.
                    return None
                if p is None:
                    vb = self.getViewWidget()
                    if vb is None:
                        return None
                    else:
                        self._viewBox = weakref.ref(vb)
                        break
                if hasattr(p, 'implements') and p.implements('ViewBox'):
                    self._viewBox = weakref.ref(p)
                    break
        return self._viewBox()  ## If we made it this far, _viewBox is definitely not None

    def forgetViewBox(self):
        self._viewBox = None
    '''

    '''
    def viewTransform(self):
        """Return the transform that maps from local coordinates to the item's ViewBox coordinates
        If there is no ViewBox, return the scene transform.
        Returns None if the item does not have a view."""
        view = self.getViewBox()
        if view is None:
            return None
        try:
            if view.implements('ViewBox'):
                tr = self.itemTransform(view.innerSceneItem())
                if isinstance(tr, tuple):
                    tr = tr[0]  # difference between pyside and pyqt
                return tr
        except:
            pass

        return self.sceneTransform()
    
    
    def viewRect(self):
        """Return the bounds (in item coordinates) of this item's ViewBox or GraphicsWidget"""
        view = self.getViewBox()
        if view is None:
            return None
        bounds = self.mapRectFromView(view.viewRect())
        if bounds is None:
            return None

        bounds = bounds.normalized()

        return bounds

    def pixelSize(self):
        vt = self.deviceTransform()
        if vt is None:
            return 0
        vt = fn.invertQTransform(vt)
        p = vt.map(QtCore.QPointF(1.0, 1.0))
        return p.x(), p.y()

    def pixelWidth(self):
        ## deprecated
        vt = self.deviceTransform()
        if vt is None:
            return 0
        vt = fn.invertQTransform(vt)
        return vt.map(QtCore.QLineF(0, 0, 1, 0)).length()

    def pixelHeight(self):
        ## deprecated
        vt = self.deviceTransform()
        if vt is None:
            return 0
        vt = fn.invertQTransform(vt)
        return vt.map(QtCore.QLineF(0, 0, 0, 1)).length()

    def mapToView(self, obj):
        vt = self.viewTransform()
        if vt is None:
            return None
        return vt.map(obj)

    def mapRectToView(self, obj):
        vt = self.viewTransform()
        if vt is None:
            return None
        return vt.mapRect(obj)

    def mapFromView(self, obj):
        vt = self.viewTransform()
        if vt is None:
            return None
        vt = fn.invertQTransform(vt)
        return vt.map(obj)

    def mapRectFromView(self, obj):
        vt = self.viewTransform()
        if vt is None:
            return None
        vt = fn.invertQTransform(vt)
        return vt.mapRect(obj)
    '''

    '''
    def pos(self):
        return Point(self._qtBaseClass.pos(self))
    '''

    '''
    def viewPos(self):
        return self.mapToView(self.mapFromParent(self.pos()))
    '''

    '''
    def parentItem(self):
        ## PyQt bug -- some items are returned incorrectly.
        return GraphicsScene.translateGraphicsItem(self._qtBaseClass.parentItem(self))
    '''

    '''
    def setParentItem(self, parent):
        ## Workaround for Qt bug: https://bugreports.qt-project.org/browse/QTBUG-18616
        if parent is not None:
            pscene = parent.scene()
            if pscene is not None and self.scene() is not pscene:
                pscene.addItem(self)
        return self._qtBaseClass.setParentItem(self, parent)
    '''

    '''
    def childItems(self):
        ## PyQt bug -- some child items are returned incorrectly.
        return list(map(GraphicsScene.translateGraphicsItem, self._qtBaseClass.childItems(self)))
    '''

    '''
    def sceneTransform(self):
        ## Qt bug: do no allow access to sceneTransform() until
        ## the item has a scene.

        if self.scene() is None:
            return self.transform()
        else:
            return self._qtBaseClass.sceneTransform(self)
    '''

    '''
    def parentIsChanged(self):
        # The function was named parentChanged() bit it was the name of a signal. So the function has been renamed.
        """Called when the item's parent has changed.
        This method handles connecting / disconnecting from ViewBox signals
        to make sure viewRangeChanged works properly. It should generally be
        extended, not overridden."""
        self._updateView()
    '''

    '''
    def _updateView(self):
        ## called to see whether this item has a new view to connect to
        ## NOTE: This is called from GraphicsObject.itemChange or GraphicsWidget.itemChange.

        ## It is possible this item has moved to a different ViewBox or widget;
        ## clear out previously determined references to these.
        self.forgetViewBox()
        self.forgetViewWidget()

        ## check for this item's current viewbox or view widget
        view = self.getViewBox()

        oldView = None
        if self._connectedView is not None:
            oldView = self._connectedView()

        if view is oldView:
            #print "  already have view", view
            return

        ## disconnect from previous view
        if oldView is not None:
            for signal, slot in [('sigRangeChanged', self.viewRangeChanged),
                                 ('sigDeviceRangeChanged', self.viewRangeChanged),
                                 ('sigTransformChanged', self.viewTransformChanged),
                                 ('sigDeviceTransformChanged', self.viewTransformChanged)]:
                try:
                    getattr(oldView, signal).disconnect(slot)
                except (TypeError, AttributeError, RuntimeError):
                    # TypeError and RuntimeError are from pyqt and pyside, respectively
                    pass

            self._connectedView = None

        ## connect to new view
        if view is not None:
            #print "connect:", self, view
            if hasattr(view, 'sigDeviceRangeChanged'):
                # connect signals from GraphicsView
                view.sigDeviceRangeChanged.connect(self.viewRangeChanged)
                view.sigDeviceTransformChanged.connect(self.viewTransformChanged)
            else:
                # connect signals from ViewBox
                view.sigRangeChanged.connect(self.viewRangeChanged)
                view.sigTransformChanged.connect(self.viewTransformChanged)
            self._connectedView = weakref.ref(view)
            self.viewRangeChanged(view.viewRange())
            self.viewTransformChanged()

        ## inform children that their view might have changed
        self._replaceView(oldView)

        self.viewChanged()
    '''

    '''
    def viewChanged(self):
        """Called when this item's view has changed
        (ie, the item has been added to or removed from a ViewBox)"""
        pass
    '''

    '''
    def _replaceView(self, oldView, item=None):
        if item is None:
            item = self
        for child in item.childItems():
            if isinstance(child, GraphicsItem):
                if child.getViewBox() is oldView:
                    child._updateView()
                        #self._replaceView(oldView, child)
            else:
                self._replaceView(oldView, child)
    '''

    '''
    def viewRangeChanged(self):
        """
        Called whenever the view coordinates of the ViewBox containing this item have changed.
        """
        pass

    def viewTransformChanged(self):
        """
        Called whenever the transformation matrix of the view has changed.
        (eg, the view range has changed or the view was resized)
        """
        pass
    '''

    '''
    def informViewBoundsChanged(self):
        """
        Inform this item's container ViewBox that the bounds of this item have changed.
        This is used by ViewBox to react if auto-range is enabled.
        """
        view = self.getViewBox()
        if view is not None and hasattr(view, 'implements') and view.implements('ViewBox'):
            view.itemBoundsChanged(self)  ## inform view so it can update its range if it wants
    '''

    '''
    def childrenShape(self):
        """Return the union of the shapes of all descendants of this item in local coordinates."""
        shapes = [self.mapFromItem(c, c.shape()) for c in self.allChildItems()]
        return reduce(operator.add, shapes)
    '''

    '''
    def allChildItems(self, root=None):
        """Return list of the entire item tree descending from this item."""
        return list(map(GraphicsScene.translateGraphicsItem, self._qtBaseClass.allChildItems(self, root=root)))
    '''

    '''
    def setExportMode(self, export, opts=None):
        """
        This method is called by exporters to inform items that they are being drawn for export
        with a specific set of options. Items access these via self._exportOptions.
        When exporting is complete, _exportOptions is set to False.
        """
        if opts is None:
            opts = {}
        if export:
            self._exportOpts = opts
            #if 'antialias' not in opts:
                #self._exportOpts['antialias'] = True
        else:
            self._exportOpts = False
    '''

    def getContextMenus(self, event):
        return [self.getMenu()] if hasattr(self, "getMenu") else []
