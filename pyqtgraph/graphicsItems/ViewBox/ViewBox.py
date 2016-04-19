import weakref
import sys
from copy import deepcopy
import numpy as np
from ...Qt import QtGui, QtCore
from ...python2_3 import sortList, basestring, cmp
from ...Point import Point
from ... import functions as fn
from .. ItemGroup import ItemGroup
from .. GraphicsWidget import GraphicsWidget
from ... import debug as debug
from ... import getConfigOption
from ...Qt import isQObjectAlive
from ...QtNativeUtils import ViewBoxBase, ChildGroup, Range, Point
from ..GraphicsItem import GraphicsItem
from PyQt4.Qt import Qt

__all__ = ['ViewBox']

'''
class WeakList(object):

    def __init__(self):
        self._items = []

    def append(self, obj):
        #Add backwards to iterate backwards (to make iterating more efficient on removal).
        self._items.insert(0, weakref.ref(obj))

    def __iter__(self):
        i = len(self._items)-1
        while i >= 0:
            ref = self._items[i]
            d = ref()
            if d is None:
                del self._items[i]
            else:
                yield d
            i -= 1
'''

'''
class ChildGroup(ItemGroup):

    def __init__(self, parent):
        ItemGroup.__init__(self, parent)

        # Used as callback to inform ViewBox when items are added/removed from
        # the group.
        # Note 1: We would prefer to override itemChange directly on the
        #         ViewBox, but this causes crashes on PySide.
        # Note 2: We might also like to use a signal rather than this callback
        #         mechanism, but this causes a different PySide crash.
        self.itemsChangedListeners = WeakList()

        # excempt from telling view when transform changes
        self._GraphicsObject__inform_view_on_change = False

    def itemChange(self, change, value):
        ret = ItemGroup.itemChange(self, change, value)
        if change == self.ItemChildAddedChange or change == self.ItemChildRemovedChange:
            try:
                itemsChangedListeners = self.itemsChangedListeners
            except AttributeError:
                # It's possible that the attribute was already collected when the itemChange happened
                # (if it was triggered during the gc of the object).
                pass
            else:
                for listener in itemsChangedListeners:
                    listener.itemsChanged()
        return ret
'''




class ViewBox(ViewBoxBase):
    """
    **Bases:** :class:`GraphicsWidget <pyqtgraph.GraphicsWidget>`

    Box that allows internal scaling/panning of children by mouse drag.
    This class is usually created automatically as part of a
    :class:`PlotItem <pyqtgraph.PlotItem>` or :class:`Canvas <pyqtgraph.canvas.Canvas>`
    or with :func:`GraphicsLayout.addViewBox() <pyqtgraph.GraphicsLayout.addViewBox>`.

    Features:

    * Scaling contents by mouse or auto-scale when contents change
    * View linking--multiple views display the same data ranges
    * Configurable by context menu
    * Item coordinate mapping methods

    """

    #sigYRangeChanged = QtCore.Signal(object, object)
    #sigXRangeChanged = QtCore.Signal(object, object)
    #sigRangeChangedManually = QtCore.Signal(object, object)
    #sigRangeChanged = QtCore.Signal(object, object)
    #sigActionPositionChanged = QtCore.Signal(object)
    #sigStateChanged = QtCore.Signal(object)
    #sigTransformChanged = QtCore.Signal(object)
    #sigResized = QtCore.Signal(object)

    ## mouse modes
    #PanMode = 3
    #RectMode = 1

    ## axes
    #XAxis = 0
    #YAxis = 1
    #XYAxes = 2

    ## for linking views together
    NamedViews = weakref.WeakValueDictionary()   # name: ViewBox
    AllViews = weakref.WeakKeyDictionary()       # ViewBox: None

    _qtBaseClass = ViewBoxBase

    def __init__(self, parent=None, border=QtGui.QPen(Qt.NoPen), lockAspect=0.0, enableMouse=True, invertY=False, enableMenu=True, name=None, invertX=False):
        """
        ==============  =============================================================
        **Arguments:**
        *parent*        (QGraphicsWidget) Optional parent widget
        *border*        (QPen) Do draw a border around the view, give any
                        single argument accepted by :func:`mkPen <pyqtgraph.mkPen>`
        *lockAspect*    (False or float) The aspect ratio to lock the view
                        coorinates to. (or False to allow the ratio to change)
        *enableMouse*   (bool) Whether mouse can be used to scale/pan the view
        *invertY*       (bool) See :func:`invertY <pyqtgraph.ViewBox.invertY>`
        *invertX*       (bool) See :func:`invertX <pyqtgraph.ViewBox.invertX>`
        *enableMenu*    (bool) Whether to display a context menu when
                        right-clicking on the ViewBox background.
        *name*          (str) Used to register this ViewBox so that it appears
                        in the "Link axis" dropdown inside other ViewBox
                        context menus. This allows the user to manually link
                        the axes of any other view to this one.
        ==============  =============================================================
        """

        ViewBoxBase.__init__(self, parent=parent, wFlags=Qt.Widget, border=border, lockAspect=lockAspect, invertX=invertX, invertY=invertY, enableMouse=enableMouse)

        #GraphicsItem.__init__(self)
        self.name = None
        #self.linksBlocked = False
        #self.addedItems = []

        #self._lastScene = None  ## stores reference to the last known scene this view was a part of.

        self.state = {

            ## separating targetRange and viewRange allows the view to be resized
            ## while keeping all previously viewed contents visible
            #'targetRange': [Point(0,1), Point(0,1)],   ## child coord. range visible [[xmin, xmax], [ymin, ymax]]
            #'viewRange': [Point(0,1), Point(0,1)],     ## actual range viewed

            #'aspectLocked': 0.0, #False,    ## False if aspect is unlocked, otherwise float specifies the locked ratio.
            #'autoRange': [True, True],  ## False if auto range is disabled,
                                          ## otherwise float gives the fraction of data that is visible
            #'autoPan': [False, False],         ## whether to only pan (do not change scaling) when auto-range is enabled
            #'autoVisibleOnly': [False, False], ## whether to auto-range only to the visible portion of a plot
            #'linkedViews': [None, None],  ## may be None, "viewName", or weakref.ref(view)
                                          ## a name string indicates that the view *should* link to another, but no view with that name exists yet.

            #'mouseEnabled': [enableMouse, enableMouse],
            #'mouseMode': ViewBox.PanMode if getConfigOption('leftButtonPan') else ViewBox.RectMode,
            'enableMenu': enableMenu,
            #'wheelScaleFactor': -1.0 / 8.0,

            # Limits
            #'limits': {
            #    'xLimits': [None, None],   # Maximum and minimum visible X values
            #    'yLimits': [None, None],   # Maximum and minimum visible Y values
            #    'xRange': [None, None],   # Maximum and minimum X range
            #    'yRange': [None, None],   # Maximum and minimum Y range
            #    }

        }

        self._updatingRange = False  ## Used to break recursive loops. See updateAutoRange.
        #self._itemBoundsCache = weakref.WeakKeyDictionary()

        self.locateGroup = None  ## items displayed when using ViewBox.locate(item)

        ## childGroup is required so that ViewBox has local coordinates similar to device coordinates.
        ## this is a workaround for a Qt + OpenGL bug that causes improper clipping
        ## https://bugreports.qt.nokia.com/browse/QTBUG-23723
        '''
        self.childGroup = ChildGroup(self)
        self.childGroup.addListener(self)
        self.setInnerSceneItem(self.childGroup)
        '''

        #self.background = QtGui.QGraphicsRectItem(self.rect())
        #self.background.setParentItem(self)
        #self.background.setZValue(-1e6)
        #self.background.setPen(fn.mkPen(None))
        #self.updateBackground()

        #self.useLeftButtonPan = pyqtgraph.getConfigOption('leftButtonPan') # normally use left button to pan
        # this also enables capture of keyPressEvents.

        ## Make scale box that is shown when dragging on the view
        '''
        self.rbScaleBox = QtGui.QGraphicsRectItem(0, 0, 1, 1)
        self.rbScaleBox.setPen(fn.mkPen((255,255,100), width=1))
        self.rbScaleBox.setBrush(fn.mkBrush(255,255,0,100))
        self.rbScaleBox.setZValue(1e9)
        self.rbScaleBox.hide()
        self.addItem(self.rbScaleBox, ignoreBounds=True)
        '''

        ## show target rect for debugging
        #self.target = QtGui.QGraphicsRectItem(0, 0, 1, 1)
        #self.target.setPen(fn.mkPen('r'))
        #self.target.setParentItem(self)
        #self.target.hide()

        #self.axHistory = [] # maintain a history of zoom locations
        #self.axHistoryPointer = -1 # pointer into the history. Allows forward/backward movement, not just "undo"

        #self.setAspectLocked(lockAspect)

        #self.border = fn.mkPen(border)

        self.menu = QtGui.QMenu() #ViewBoxMenu(self)

        self.register(name)
        if name is None:
            self.updateViewLists()

    def register(self, name):
        """
        Add this ViewBox to the registered list of views.

        This allows users to manually link the axes of any other ViewBox to
        this one. The specified *name* will appear in the drop-down lists for
        axis linking in the context menus of all other views.

        The same can be accomplished by initializing the ViewBox with the *name* attribute.
        """
        ViewBox.AllViews[self] = None
        if self.name is not None:
            del ViewBox.NamedViews[self.name]
        self.name = name
        if name is not None:
            ViewBox.NamedViews[name] = self
            ViewBox.updateAllViewLists()
            sid = id(self)
            self.destroyed.connect(lambda: ViewBox.forgetView(sid, name) if (ViewBox is not None and 'sid' in locals() and 'name' in locals()) else None)
            #self.destroyed.connect(self.unregister)

    def unregister(self):
        """
        Remove this ViewBox from the list of linkable views. (see :func:`register() <pyqtgraph.ViewBox.register>`)
        """
        del ViewBox.AllViews[self]
        if self.name is not None:
            del ViewBox.NamedViews[self.name]

    def close(self):
        self.clear()
        self.unregister()

    def implements(self, interface):
        return interface == 'ViewBox'

    '''
    def checkSceneChange(self):
        # ViewBox needs to receive sigPrepareForPaint from its scene before
        # being painted. However, we have no way of being informed when the
        # scene has changed in order to make this connection. The usual way
        # to do this is via itemChange(), but bugs prevent this approach
        # (see above). Instead, we simply check at every paint to see whether
        # (the scene has changed.
        scene = self.scene()
        if scene == self._lastScene:
            return
        if self._lastScene is not None and hasattr(self.lastScene, 'sigPrepareForPaint'):
            self._lastScene.sigPrepareForPaint.disconnect(self.prepareForPaint)
        if scene is not None and hasattr(scene, 'sigPrepareForPaint'):
            scene.sigPrepareForPaint.connect(self.prepareForPaint)
        self.prepareForPaint()
        self._lastScene = scene

    def prepareForPaint(self):
        # don't check whether auto range is enabled here--only check when setting dirty flag.
        if self.autoRangeNeedsUpdate():  # and autoRangeEnabled:
            self.updateAutoRange()
        if self.matrixNeedsUpdate():
            self.updateMatrix()
    '''

    def getState(self, copy=True):
        """Return the current state of the ViewBox.
        Linked views are always converted to view names in the returned state."""
        state = self.state.copy()
        views = []
        for v in state['linkedViews']:
            if isinstance(v, weakref.ref):
                v = v()
            if v is None or isinstance(v, basestring):
                views.append(v)
            else:
                views.append(v.name)
        state['linkedViews'] = views
        state['xInverted'] = self.xInverted()
        state['yInverted'] = self.yInverted()
        state['viewRange'] = self.viewRange()
        state['background'] = self.backgroundColor()
        state['targetRange'] = self.targetRange()
        aspectLocked = self.aspectLocked()
        state['aspectLocked'] = aspectLocked if aspectLocked != 0.0 else False
        state['autoRange'] = self.autoRangeEnabled()
        state['autoPan'] = self.autoPan()
        if copy:
            return deepcopy(state)
        else:
            return state

    def setState(self, state):
        """Restore the state of this ViewBox.
        (see also getState)"""
        state = state.copy()
        self.setXLink(state['linkedViews'][0])
        self.setYLink(state['linkedViews'][1])
        del state['linkedViews']

        self.state.update(state)
        if 'xInverted' in state:
            self.invertX(state['xInverted'])
        if 'yInverted' in state:
            self.invertY(state['yInverted'])

        if 'viewRange' in state:
            viewRange = state['viewRange']
            self.setViewRange(viewRange[0], viewRange[1])

        if 'background' in state:
            b = state['background']
            color = b if b is not None else QtGui.QColor(0, 0, 0, 0)
            self.setBackgroundColor(color)

        if 'aspectLocked' in state:
            aspectLocked = state['aspectLocked']
            ratio = aspectLocked if aspectLocked is not False else 1.0
            self.setAspectLocked(locked=aspectLocked != 0.0, ratio=ratio)

        if 'autoRange' in state:
            are = state['autoRange']
            self.setAutoRangeEnabled(are[0], are[1])

        if 'autoPan' in state:
            ap = state['autoPan']
            self.setAutoPan(ap[0], ap[1])

        self.updateViewRange()
        self.sigStateChanged.emit(self)

    '''
    def setMouseMode(self, mode):
        """
        Set the mouse interaction mode. *mode* must be either ViewBox.PanMode or ViewBox.RectMode.
        In PanMode, the left mouse button pans the view and the right button scales.
        In RectMode, the left button draws a rectangle which updates the visible region (this mode is more suitable for single-button mice)
        """
        if mode not in [ViewBox.PanMode, ViewBox.RectMode]:
            raise Exception("Mode must be ViewBox.PanMode or ViewBox.RectMode")
        self.state['mouseMode'] = mode
        self.sigStateChanged.emit(self)
    '''

    def setLeftButtonAction(self, mode='rect'):  ## for backward compatibility
        raise RuntimeWarning('ViewBox.setLeftButtonAction is deprecated')
        if mode.lower() == 'rect':
            self.setMouseMode(ViewBox.RectMode)
        elif mode.lower() == 'pan':
            self.setMouseMode(ViewBox.PanMode)
        else:
            raise Exception('graphicsItems:ViewBox:setLeftButtonAction: unknown mode = %s (Options are "pan" and "rect")' % mode)

    '''
    def innerSceneItem(self):
        return self.childGroup
    '''

    '''
    def setMouseEnabled(self, x=None, y=None):
        """
        Set whether each axis is enabled for mouse interaction. *x*, *y* arguments must be True or False.
        This allows the user to pan/scale one axis of the view while leaving the other axis unchanged.
        """
        if x is not None:
            self.state['mouseEnabled'][0] = x
        if y is not None:
            self.state['mouseEnabled'][1] = y
        self.sigStateChanged.emit(self)

    def mouseEnabled(self):
        return self.state['mouseEnabled'][:]
    '''

    def setMenuEnabled(self, enableMenu=True):
        self.state['enableMenu'] = enableMenu
        self.sigStateChanged.emit(self)

    def menuEnabled(self):
        return self.state.get('enableMenu', True)

    '''
    def addItem(self, item, ignoreBounds=False):
        """
        Add a QGraphicsItem to this view. The view will include this item when determining how to set its range
        automatically unless *ignoreBounds* is True.
        """
        if item.zValue() < self.zValue():
            item.setZValue(self.zValue()+1)
        scene = self.scene()
        if scene is not None and scene is not item.scene():
            scene.addItem(item)  ## Necessary due to Qt bug: https://bugreports.qt-project.org/browse/QTBUG-18616
        item.setParentItem(self.getChildGroup())
        if not ignoreBounds:
            self.addedItems.append(item)
        self.updateAutoRange()
        #print "addItem:", item, item.boundingRect()

    def removeItem(self, item):
        """Remove an item from this view."""
        try:
            self.addedItems.remove(item)
        except:
            pass
        self.scene().removeItem(item)
        self.updateAutoRange()

    def clear(self):
        for i in self.addedItems[:]:
            self.removeItem(i)
        for ch in self.getChildGroup().childItems():
            ch.setParentItem(None)
    '''
    '''
    def resizeEvent(self, ev):
        self.linkedXChanged()
        self.linkedYChanged()
        self.updateAutoRange()
        self.updateViewRange()
        self.setMatrixNeedsUpdate(True)
        self.sigStateChanged.emit(self)
        self.updateBackground()
        self.sigResized.emit()
    '''
    '''
    def viewRect(self):
        """Return a QRectF bounding the region visible within the ViewBox"""
        try:
            viewRange = self.viewRange()
            print viewRange
            vr0 = viewRange[0]
            vr1 = viewRange[1]
            r = QtCore.QRectF(vr0[0], vr1[0], vr0[1]-vr0[0], vr1[1] - vr1[0])
            print r
            return r
        except:
            print("make qrectf failed:", self.viewRange())
            raise
    '''
    '''
    def targetRect(self):
        """
        Return the region which has been requested to be visible.
        (this is not necessarily the same as the region that is *actually* visible--
        resizing and aspect ratio constraints can cause targetRect() and viewRect() to differ)
        """
        try:
            tr = self.targetRange()
            tr0 = tr[0]
            tr1 = tr[1]
            return QtCore.QRectF(tr0[0], tr1[0], tr0[1]-tr0[0], tr1[1] - tr1[0])
        except:
            print("make qrectf failed:", tr)
            raise
    '''

    '''
    def _resetTarget(self):
        # Reset target range to exactly match current view range.
        # This is used during mouse interaction to prevent unpredictable
        # behavior (because the user is unaware of targetRange).
        if self.aspectLocked() == 0.0:  # (interferes with aspect locking)
            viewRange = self.viewRange()
            self.setTargetRange(viewRange[0], viewRange[1])
    '''

    # setRange(const QPointF& xRange=QPointF(), const QPointF& yRange=QPointF(), const double padding=std::numeric_limits::quiet_NaN(), const bool disableAutoRange=true)
    # setRange(const QRectF& rect=QPointF(), const double padding=std::numeric_limits::quiet_NaN(), const bool disableAutoRange=true)

    # setRange(const QPointF& xRange=QPointF(), const QPoint& yRange=QPointF(), const bool disableAutoRange=true)
    # setRange(const QRectF& rect=QPointF(), const bool disableAutoRange=true)

    '''
    def setRange(self, rect=None, xRange=None, yRange=None, padding=None, update=True, disableAutoRange=True):
        """
        Set the visible range of the ViewBox.
        Must specify at least one of *rect*, *xRange*, or *yRange*.

        ================== =====================================================================
        **Arguments:**
        *rect*             (QRectF) The full range that should be visible in the view box.
        *xRange*           (min,max) The range that should be visible along the x-axis.
        *yRange*           (min,max) The range that should be visible along the y-axis.
        *padding*          (float) Expand the view by a fraction of the requested range.
                           By default, this value is set between 0.02 and 0.1 depending on
                           the size of the ViewBox.
        *update*           (bool) If True, update the range of the ViewBox immediately.
                           Otherwise, the update is deferred until before the next render.
        *disableAutoRange* (bool) If True, auto-ranging is disabled. Otherwise, it is left
                           unchanged.
        ================== =====================================================================

        """
        # Update is ignored

        print 'setRange', rect, xRange, yRange, padding, update, disableAutoRange
        print
        print

        super(ViewBox, self).setRange(rect=rect, xRange=xRange, yRange=yRange, padding=padding, update=update, disableAutoRange=disableAutoRange)
        return

        changes = {}   # axes
        setRequested = [False, False]

        if rect is not None:
            changes = {0: [rect.left(), rect.right()], 1: [rect.top(), rect.bottom()]}
            setRequested = [True, True]
        if xRange is not None:
            changes[0] = xRange
            setRequested[0] = True
        if yRange is not None:
            changes[1] = yRange
            setRequested[1] = True

        if len(changes) == 0:
            print(rect)
            raise Exception("Must specify at least one of rect, xRange, or yRange. (gave rect=%s)" % str(type(rect)))

        # Update axes one at a time
        changed = [False, False]
        for ax, range in changes.items():
            mn = min(range)
            mx = max(range)

            # If we requested 0 range, try to preserve previous scale.
            # Otherwise just pick an arbitrary scale.
            if mn == mx:
                dy = self.viewRange()[ax]
                dy = dy[1] - dy[0]
                if dy == 0:
                    dy = 1
                mn -= dy*0.5
                mx += dy*0.5
                xpad = 0.0

            # Make sure no nan/inf get through
            if not all(np.isfinite([mn, mx])):
                raise Exception("Cannot set range [%s, %s]" % (str(mn), str(mx)))

            # Apply padding
            if padding is None:
                xpad = self.suggestPadding(ax)
            else:
                xpad = padding
            p = (mx-mn) * xpad
            mn -= p
            mx += p

            # Set target range
            curTargetRange = self.targetRange()
            if curTargetRange[ax] != Range(mn, mx):
                curTargetRange[ax] = Range(mn, mx)
                self.setTargetRange(Range(curTargetRange[0]), Range(curTargetRange[1]))
                changed[ax] = True

        # Update viewRange to match targetRange as closely as possible while
        # accounting for aspect ratio constraint
        lockX, lockY = setRequested
        if lockX and lockY:
            lockX = False
            lockY = False
        self.updateViewRange(lockX, lockY)

        # Disable auto-range for each axis that was requested to be set
        if disableAutoRange:
            if setRequested[0]:
                self.enableAutoRange(ViewBox.XAxis, enable=False)
            if setRequested[1]:
                self.enableAutoRange(ViewBox.YAxis, enable=False)
            changed.append(True)

        # If nothing has changed, we are done.
        if any(changed):

            self.sigStateChanged.emit(self)

            # Update target rect for debugging -- Ignored during porting
            # if self.target.isVisible():
            #     self.target.setRect(self.mapRectFromItem(self.getChildGroup(), self.targetRect()))

        # If ortho axes have auto-visible-only, update them now
        # Note that aspect ratio constraints and auto-visible probably do not work together..
        autoVisibleOnly = self.autoVisible()
        if changed[0] and autoVisibleOnly[1] and (self.autoRangeEnabled()[0] is not False):
            self.setAutoRangeNeedsUpdate(True)
        elif changed[1] and autoVisibleOnly[0] and (self.autoRangeEnabled()[1] is not False):
            self.setAutoRangeNeedsUpdate(True)
    '''

    '''
    def setYRange(self, min, max, padding=None, update=True):
        """
        Set the visible Y range of the view to [*min*, *max*].
        The *padding* argument causes the range to be set larger by the fraction specified.
        (by default, this value is between 0.02 and 0.1 depending on the size of the ViewBox)
        """
        self.setRange(yRange=[min, max], update=update, padding=padding)

    def setXRange(self, min, max, padding=None, update=True):
        """
        Set the visible X range of the view to [*min*, *max*].
        The *padding* argument causes the range to be set larger by the fraction specified.
        (by default, this value is between 0.02 and 0.1 depending on the size of the ViewBox)
        """
        self.setRange(xRange=[min, max], update=update, padding=padding)
    '''

    def autoRange(self, padding=None, items=None, item=None):
        """
        Set the range of the view box to make all children visible.
        Note that this is not the same as enableAutoRange, which causes the view to
        automatically auto-range whenever its contents are changed.

        ==============  ============================================================
        **Arguments:**
        padding         The fraction of the total data range to add on to the final
                        visible range. By default, this value is set between 0.02
                        and 0.1 depending on the size of the ViewBox.
        items           If specified, this is a list of items to consider when
                        determining the visible range.
        ==============  ============================================================
        """
        if item is None:
            bounds = self.childrenBoundingRect(items=items)
        else:
            print("Warning: ViewBox.autoRange(item=__) is deprecated. Use 'items' argument instead.")
            bounds = self.mapFromItemToView(item, item.boundingRect()).boundingRect()

        if bounds is not None:
            self.setRange(bounds, padding=padding)

    '''
    def suggestPadding(self, axis):
        l = self.width() if axis==0 else self.height()
        if l > 0:
            padding = np.clip(1./(l**0.5), 0.02, 0.1)
        else:
            padding = 0.02
        return padding
    '''

    '''
    def setLimits(self, **kwds):
        """
        Set limits that constrain the possible view ranges.

        **Panning limits**. The following arguments define the region within the
        viewbox coordinate system that may be accessed by panning the view.

        =========== ============================================================
        xMin        Minimum allowed x-axis value
        xMax        Maximum allowed x-axis value
        yMin        Minimum allowed y-axis value
        yMax        Maximum allowed y-axis value
        =========== ============================================================

        **Scaling limits**. These arguments prevent the view being zoomed in or
        out too far.

        =========== ============================================================
        minXRange   Minimum allowed left-to-right span across the view.
        maxXRange   Maximum allowed left-to-right span across the view.
        minYRange   Minimum allowed top-to-bottom span across the view.
        maxYRange   Maximum allowed top-to-bottom span across the view.
        =========== ============================================================

        Added in version 0.9.9
        """
        update = False
        allowed = ['xMin', 'xMax', 'yMin', 'yMax', 'minXRange', 'maxXRange', 'minYRange', 'maxYRange']
        for kwd in kwds:
            if kwd not in allowed:
                raise ValueError("Invalid keyword argument '%s'." % kwd)
        #for kwd in ['xLimits', 'yLimits', 'minRange', 'maxRange']:
            #if kwd in kwds and self.state['limits'][kwd] != kwds[kwd]:
                #self.state['limits'][kwd] = kwds[kwd]
                #update = True

        limits = self.state['limits']

        if 'xMin' in kwds:
            val = kwds['xMin']
            if val != limits['xLimits'][0]:
                limits['xLimits'][0] = val
                update = True
        if 'xMax' in kwds:
            val = kwds['xMax']
            if val != limits['xLimits'][1]:
                limits['xLimits'][1] = val
                update = True
        if 'yMin' in kwds:
            val = kwds['yMin']
            if val != limits['yLimits'][0]:
                limits['yLimits'][0] = val
                update = True
        if 'yMax' in kwds:
            val = kwds['yMax']
            if val != limits['yLimits'][1]:
                limits['yLimits'][1] = val
                update = True
        if 'minXRange' in kwds:
            val = kwds['minXRange']
            if val != limits['xRange'][0]:
                limits['xRange'][0] = val
                update = True
        if 'maxXRange' in kwds:
            val = kwds['maxXRange']
            if val != limits['xRange'][1]:
                limits['xRange'][1] = val
                update = True
        if 'minYRange' in kwds:
            val = kwds['minYRange']
            if val != limits['yRange'][0]:
                limits['yRange'][0] = val
                update = True
        if 'maxYRange' in kwds:
            val = kwds['maxYRange']
            if val != limits['yRange'][1]:
                limits['yRange'][1] = val
                update = True

        if update:
            self.updateViewRange()
    '''

    '''
    def scaleBy(self, s=None, center=None, x=None, y=None):
        """
        Scale by *s* around given center point (or center of view).
        *s* may be a Point or tuple (x, y).

        Optionally, x or y may be specified individually. This allows the other
        axis to be left unaffected (note that using a scale factor of 1.0 may
        cause slight changes due to floating-point error).
        """
        if s is not None:
            scale = Point(s)
        else:
            scale = [x, y]

        affect = [True, True]
        if scale[0] is None and scale[1] is None:
            return
        elif scale[0] is None:
            affect[0] = False
            scale[0] = 1.0
        elif scale[1] is None:
            affect[1] = False
            scale[1] = 1.0

        scale = Point(scale)

        if self.aspectLocked() != 0.0:
            scale[0] = scale[1]

        vr = self.targetRect()
        if center is None:
            center = Point(vr.center())
        else:
            center = Point(center)

        tl = center + (vr.topLeft()-center) * scale
        br = center + (vr.bottomRight()-center) * scale

        if not affect[0]:
            self.setYRange(tl.y(), br.y(), padding=0)
        elif not affect[1]:
            self.setXRange(tl.x(), br.x(), padding=0)
        else:
            self.setRange(QtCore.QRectF(tl, br), padding=0)
    '''
    '''
    def translateBy(self, t=None, x=None, y=None):
        """
        Translate the view by *t*, which may be a Point or tuple (x, y).

        Alternately, x or y may be specified independently, leaving the other
        axis unchanged (note that using a translation of 0 may still cause
        small changes due to floating-point error).
        """
        vr = self.targetRect()
        if t is not None:
            t = Point(t)
        else:
            x = x if x else 0.0
            y = y if y else 0.0
            t = Point(x, y)

        self.setRange(vr.translated(t), padding=0)
    '''
    '''
    def enableAutoRange(self, axis=None, enable=True, x=None, y=None):
        """
        Enable (or disable) auto-range for *axis*, which may be ViewBox.XAxis, ViewBox.YAxis, or ViewBox.XYAxes for both
        (if *axis* is omitted, both axes will be changed).
        When enabled, the axis will automatically rescale when items are added/removed or change their shape.
        The argument *enable* may optionally be a float (0.0-1.0) which indicates the fraction of the data that should
        be visible (this only works with items implementing a dataRange method, such as PlotDataItem).
        """
        #print "autorange:", axis, enable
        #if not enable:
            #import traceback
            #traceback.print_stack()

        # support simpler interface:
        if x is not None or y is not None:
            if x is not None:
                self.enableAutoRange(ViewBox.XAxis, x)
            if y is not None:
                self.enableAutoRange(ViewBox.YAxis, y)
            return

        if enable is True:
            enable = 1.0

        if axis is None:
            axis = ViewBox.XYAxes

        #needAutoRangeUpdate = False

        if axis == ViewBox.XYAxes or axis == 'xy':
            axes = [0, 1]
        elif axis == ViewBox.XAxis or axis == 'x':
            axes = [0]
        elif axis == ViewBox.YAxis or axis == 'y':
            axes = [1]
        else:
            raise Exception('axis argument must be ViewBox.XAxis, ViewBox.YAxis, or ViewBox.XYAxes.')

        for ax in axes:
            are = self.autoRangeEnabled()
            if are[ax] != enable:
                # If we are disabling, do one last auto-range to make sure that
                # previously scheduled auto-range changes are enacted
                if enable is False and self.autoRangeNeedsUpdate():
                    self.updateAutoRange()

                are[ax] = enable
                self.setAutoRangeEnabled(are[0], are[1])
                self.setAutoRangeNeedsUpdate(self.autoRangeNeedsUpdate() or (enable is not False))
                self.update()

        if self.autoRangeNeedsUpdate():
            self.updateAutoRange()

        self.sigStateChanged.emit(self)
    '''

    def disableAutoRange(self, axis=None):
        """Disables auto-range. (See enableAutoRange)"""
        if axis is None:
            axis = ViewBox.XYAxes
        self.enableAutoRange(axis, enable=False)

    #def autoRangeEnabled(self):
    #    return self.state['autoRange'][:]

    #def setAutoPan(self, x=None, y=None):
    #    if x is not None:
    #        self.state['autoPan'][0] = x
    #    if y is not None:
    #        self.state['autoPan'][1] = y
    #    if None not in [x,y]:
    #        self.updateAutoRange()

    #def setAutoVisible(self, x=None, y=None):
    #    if x is not None:
    #        self.state['autoVisibleOnly'][0] = x
    #        if x is True:
    #            self.state['autoVisibleOnly'][1] = False
    #    if y is not None:
    #        self.state['autoVisibleOnly'][1] = y
    #        if y is True:
    #            self.state['autoVisibleOnly'][0] = False
    #
    #    if x is not None or y is not None:
    #        self.updateAutoRange()

    def updateAutoRange(self):
        ## Break recursive loops when auto-ranging.
        ## This is needed because some items change their size in response
        ## to a view change.
        if self._updatingRange:
            return

        self._updatingRange = True
        try:
            targetRect = self.viewRange()
            if not any(self.autoRangeEnabled()):
                return

            fractionVisible = self.autoRangeEnabled()
            for i in [0, 1]:
                if type(fractionVisible[i]) is bool:
                    fractionVisible[i] = 1.0

            fractionVisible = Point(fractionVisible[0], fractionVisible[1])

            childRange = None

            autoVisibleOnly = self.autoVisible()

            order = [0, 1]
            if autoVisibleOnly[0] is True:
                order = [1, 0]

            args = {}
            for ax in order:
                if self.autoRangeEnabled()[ax] is False:
                    continue
                if autoVisibleOnly[ax]:
                    oRange = [None, None]
                    oRange[ax] = targetRect[1 - ax]
                    childRange = self.childrenBounds(frac=fractionVisible, orthoRange=oRange)

                else:
                    if childRange is None:
                        childRange = self.childrenBounds(frac=fractionVisible)

                ## Make corrections to range
                xr = childRange[ax]
                if xr is not None:
                    if self.autoPan()[ax]:
                        x = sum(xr) * 0.5
                        w2 = (targetRect[ax][1] - targetRect[ax][0]) / 2.
                        childRange[ax] = [x - w2, x + w2]
                    else:
                        padding = self.suggestPadding(ax)
                        wp = (xr[1] - xr[0]) * padding
                        childRange[ax][0] -= wp
                        childRange[ax][1] += wp
                    targetRect[ax] = childRange[ax]
                    args['xRange' if ax == 0 else 'yRange'] = targetRect[ax]
            if len(args) == 0:
                return
            args['padding'] = 0
            args['disableAutoRange'] = False

            xRange = Range(args.get('xRange', (np.nan, np.nan)))
            yRange = Range(args.get('yRange', (np.nan, np.nan)))
            padding = 0
            disableAutoRange = False

            self.setRange(xRange=xRange, yRange=yRange, padding=padding, disableAutoRange=disableAutoRange)
        except:
            import traceback
            traceback.print_exc()
        finally:
            self.setAutoRangeNeedsUpdate(False)
            self._updatingRange = False

    '''
    def setXLink(self, view):
        """Link this view's X axis to another view. (see LinkView)"""
        self.linkView(self.XAxis, view)

    def setYLink(self, view):
        """Link this view's Y axis to another view. (see LinkView)"""
        self.linkView(self.YAxis, view)
    '''

    '''
    def linkView(self, axis, view):
        """
        Link X or Y axes of two views and unlink any previously connected axes. *axis* must be ViewBox.XAxis or ViewBox.YAxis.
        If view is None, the axis is left unlinked.
        """
        if isinstance(view, basestring):
            if view == '':
                view = None
            else:
                view = ViewBox.NamedViews.get(view, view)  ## convert view name to ViewBox if possible

        if hasattr(view, 'implements') and view.implements('ViewBoxWrapper'):
            view = view.getViewBox()

        ## used to connect/disconnect signals between a pair of views
        if axis == ViewBox.XAxis:
            signal = 'sigXRangeChanged'
            slot = self.linkedXChanged
        else:
            signal = 'sigYRangeChanged'
            slot = self.linkedYChanged


        oldLink = self.linkedView(axis)
        if oldLink is not None:
            try:
                getattr(oldLink, signal).disconnect(slot)
                oldLink.sigResized.disconnect(slot)
            except (TypeError, RuntimeError):
                ## This can occur if the view has been deleted already
                pass


        if view is None or isinstance(view, basestring):
            self.state['linkedViews'][axis] = view
        else:
            self.state['linkedViews'][axis] = weakref.ref(view)
            getattr(view, signal).connect(slot)
            view.sigResized.connect(slot)
            if view.autoRangeEnabled()[axis] is not False:
                self.enableAutoRange(axis, False)
                slot()
            else:
                if self.autoRangeEnabled()[axis] is False:
                    slot()


        self.sigStateChanged.emit(self)
    '''

    #def blockLink(self, b):
    #    self.linksBlocked = b  ## prevents recursive plot-change propagation

    '''
    def linkedXChanged(self):
        ## called when x range of linked view has changed
        view = self.linkedView(0)
        self.linkedViewChanged(view, ViewBox.XAxis)

    def linkedYChanged(self):
        ## called when y range of linked view has changed
        view = self.linkedView(1)
        self.linkedViewChanged(view, ViewBox.YAxis)

    def linkedView(self, ax):
        ## Return the linked view for axis *ax*.
        ## this method _always_ returns either a ViewBox or None.
        v = self.state['linkedViews'][ax]
        if v is None or isinstance(v, basestring):
            return None
        else:
            return v()  ## dereference weakref pointer. If the reference is dead, this returns None
    '''
    '''
    def linkedViewChanged(self, view, axis):
        if self.linksBlocked() or view is None:
            return

        #print self.name, "ViewBox.linkedViewChanged", axis, view.viewRange()[axis]
        vr = view.viewRect()
        vg = view.screenGeometry()
        sg = self.screenGeometry()
        if vg is None or sg is None:
            return

        view.blockLink(True)
        try:
            if axis == ViewBox.XAxis:
                overlap = min(sg.right(), vg.right()) - max(sg.left(), vg.left())
                if overlap < min(vg.width()/3, sg.width()/3):  ## if less than 1/3 of views overlap,
                                                               ## then just replicate the view
                    x1 = vr.left()
                    x2 = vr.right()
                else:  ## views overlap; line them up
                    upp = float(vr.width()) / vg.width()
                    if self.xInverted():
                        x1 = vr.left()   + (sg.right()-vg.right()) * upp
                    else:
                        x1 = vr.left()   + (sg.x()-vg.x()) * upp
                    x2 = x1 + sg.width() * upp
                self.enableAutoRange(ViewBox.XAxis, False)
                self.setXRange(x1, x2, padding=0)
            else:
                overlap = min(sg.bottom(), vg.bottom()) - max(sg.top(), vg.top())
                if overlap < min(vg.height()/3, sg.height()/3):  ## if less than 1/3 of views overlap,
                                                                 ## then just replicate the view
                    y1 = vr.top()
                    y2 = vr.bottom()
                else:  ## views overlap; line them up
                    upp = float(vr.height()) / vg.height()
                    if self.yInverted():
                        y2 = vr.bottom() + (sg.bottom()-vg.bottom()) * upp
                    else:
                        y2 = vr.bottom() + (sg.top()-vg.top()) * upp
                    y1 = y2 - sg.height() * upp
                self.enableAutoRange(ViewBox.YAxis, False)
                self.setYRange(y1, y2, padding=0)
        finally:
            view.blockLink(False)
    '''

    '''
    def screenGeometry(self):
        """return the screen geometry of the viewbox"""
        v = self.getViewWidget()
        if v is None:
            return None
        b = self.sceneBoundingRect()
        wr = v.mapFromScene(b).boundingRect()
        pos = v.mapToGlobal(v.pos())
        wr.adjust(pos.x(), pos.y(), pos.x(), pos.y())
        return wr
    '''

    '''
    def itemsChanged(self):
        ## called when items are added/removed from self.childGroup
        self.updateAutoRange()
    '''

    #def itemBoundsChanged(self, item):
    #    #self._itemBoundsCache.pop(item, None)
    #    are = self.autoRangeEnabled()
    #    if (are[0] is not False) or (are[1] is not False):
    #        self.setAutoRangeNeedsUpdate(True)
    #        self.update()
    #    #self.updateAutoRange()
    '''
    def setAspectLocked(self, lock=True, ratio=1):
        """
        If the aspect ratio is locked, view scaling must always preserve the aspect ratio.
        By default, the ratio is set to 1; x and y both have the same scaling.
        This ratio can be overridden (xScale/yScale), or use None to lock in the current ratio.
        """

        if not lock:
            if self.aspectLocked() == 0.0:
                return
            ViewBoxBase.setAspectLocked(self, False, 0.0)
        else:
            rect = self.rect()
            vr = self.viewRect()
            if rect.height() == 0 or vr.width() == 0 or vr.height() == 0:
                currentRatio = 1.0
            else:
                currentRatio = (rect.width()/float(rect.height())) / (vr.width()/vr.height())
            if ratio is None:
                ratio = currentRatio
            if self.aspectLocked() == ratio: # nothing to change
                return
            ViewBoxBase.setAspectLocked(self, True, ratio)
            #if ratio != currentRatio:  ## If this would change the current range, do that now
            #    self.updateViewRange()

        self.updateAutoRange()
        self.updateViewRange()
        self.sigStateChanged.emit(self)
    '''
    '''
    def childTransform(self):
        """
        Return the transform that maps from child(item in the childGroup) coordinates to local coordinates.
        (This maps from inside the viewbox to outside)
        """
        if self.matrixNeedsUpdate():
            self.updateMatrix()
        m = self.getChildGroup().transform()
        #m1 = QtGui.QTransform()
        #m1.translate(self.childGroup.pos().x(), self.childGroup.pos().y())
        return m #*m1
    '''

    '''
    def mapToView(self, obj):
        """Maps from the local coordinates of the ViewBox to the coordinate system displayed inside the ViewBox"""
        m = fn.invertQTransform(self.childTransform())
        return m.map(obj)

    def mapFromView(self, obj):
        """Maps from the coordinate system displayed inside the ViewBox to the local coordinates of the ViewBox"""
        m = self.childTransform()
        return m.map(obj)

    def mapSceneToView(self, obj):
        """Maps from scene coordinates to the coordinate system displayed inside the ViewBox"""
        return self.mapToView(self.mapFromScene(obj))

    def mapViewToScene(self, obj):
        """Maps from the coordinate system displayed inside the ViewBox to scene coordinates"""
        return self.mapToScene(self.mapFromView(obj))

    def mapFromItemToView(self, item, obj):
        """Maps *obj* from the local coordinate system of *item* to the view coordinates"""
        return self.getChildGroup().mapFromItem(item, obj)
        #return self.mapSceneToView(item.mapToScene(obj))

    def mapFromViewToItem(self, item, obj):
        """Maps *obj* from view coordinates to the local coordinate system of *item*."""
        return self.getChildGroup().mapToItem(item, obj)
        #return item.mapFromScene(self.mapViewToScene(obj))

    def mapViewToDevice(self, obj):
        return self.mapToDevice(self.mapFromView(obj))

    def mapDeviceToView(self, obj):
        return self.mapToView(self.mapFromDevice(obj))
    '''
    '''
    def viewPixelSize(self):
        """Return the (width, height) of a screen pixel in view coordinates."""
        o = self.mapToView(Point(0,0))
        px, py = [Point(self.mapToView(v) - o) for v in self.pixelVectors()]
        return (px.length(), py.length())
    '''
    '''
    def itemBoundingRect(self, item):
        """Return the bounding rect of the item in view coordinates"""
        return self.mapSceneToView(item.sceneBoundingRect()).boundingRect()
    '''

    #def viewScale(self):
        #vr = self.viewRect()
        ##print "viewScale:", self.range
        #xd = vr.width()
        #yd = vr.height()
        #if xd == 0 or yd == 0:
            #print "Warning: 0 range in view:", xd, yd
            #return np.array([1,1])

        ##cs = self.canvas().size()
        #cs = self.boundingRect()
        #scale = np.array([cs.width() / xd, cs.height() / yd])
        ##print "view scale:", scale
        #return scale
    '''
    def wheelEvent(self, ev, axis=None):
        mask = np.array(self.mouseEnabled(), dtype=np.float)
        if axis is not None and axis >= 0 and axis < len(mask):
            mv = mask[axis]
            mask[:] = 0
            mask[axis] = mv
        s = ((mask * 0.02) + 1) ** (ev.delta() * self.wheelScaleFactor()) # actual scaling factor
        s = Point(s)

        center = Point(fn.invertQTransform(self.getChildGroup().transform()).map(ev.pos()))

        self._resetTarget()
        self.scaleBy(s, center)
        s = self.mouseEnabled()
        self.sigRangeChangedManually.emit(s[0], s[1])
        ev.accept()
    '''

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton and self.menuEnabled():
            ev.accept()
            self.raiseContextMenu(ev)

    def raiseContextMenu(self, ev):
        menu = self.getMenu(ev)
        self.scene().addParentContextMenus(self, menu, ev)
        menu.popup(ev.screenPos())

    def getMenu(self, ev):
        return self.menu

    def getContextMenus(self, event):
        return self.menu.actions() if self.menuEnabled() else []

    '''
    def mouseDragEvent(self, ev, axis=None):
        ## if axis is specified, event will only affect that axis.
        ev.accept()  ## we accept all buttons

        pos = ev.pos()
        lastPos = ev.lastPos()
        dif = lastPos - pos

        ## Ignore axes if mouse is disabled
        mouseEnabled = np.array(self.mouseEnabled(), dtype=np.float)
        mask = mouseEnabled.copy()
        if axis is not None:
            mask[1-axis] = 0.0

        ## Scale or translate based on mouse button
        if ev.button() & (QtCore.Qt.LeftButton | QtCore.Qt.MidButton):
            if self.mouseMode() == ViewBox.RectMode:
                if ev.isFinish():  ## This is the final move in the drag; change the view scale now
                    self.hideScaleBox()
                    ax = QtCore.QRectF(Point(ev.buttonDownPos(ev.button())), Point(pos))
                    ax = self.getChildGroup().mapRectFromParent(ax)
                    self.showAxRect(ax)
                    self.addToHistory(ax)
                else:
                    ## update shape of scale box
                    self.updateScaleBox(ev.buttonDownPos(), ev.pos())
            else:
                tr = self.getChildGroup().transform()
                tr = fn.invertQTransform(tr)
                tr = tr.map(Point(dif)*Point(mask)) - tr.map(Point(0, 0))
                x = tr.x() if mask[0] == 1 else 0.0
                y = tr.y() if mask[1] == 1 else 0.0

                self._resetTarget()
                if x != 0.0 or y != 0.0:
                    self.translateBy(x=x, y=y)
                s = self.mouseEnabled()
                self.sigRangeChangedManually.emit(s[0], s[1])
        elif ev.button() & QtCore.Qt.RightButton:
            if self.aspectLocked() != 0.0:
                mask[0] = 0

            dif = ev.screenPos() - ev.lastScreenPos()
            dif = np.array([dif.x(), dif.y()])
            dif[0] *= -1
            s = ((mask * 0.02) + 1) ** dif

            tr = self.getChildGroup().transform()
            tr = fn.invertQTransform(tr)

            x = s[0] if mouseEnabled[0] == 1 else 0.0
            y = s[1] if mouseEnabled[1] == 1 else 0.0

            center = Point(tr.map(ev.buttonDownPos(QtCore.Qt.RightButton)))
            self._resetTarget()
            self.scaleBy(x=x, y=y, center=center)
            s = self.mouseEnabled()
            self.sigRangeChangedManually.emit(s[0], s[1])
    '''
    '''
    def keyPressEvent(self, ev):
        """
        This routine should capture key presses in the current view box.
        Key presses are used only when mouse mode is RectMode
        The following events are implemented:
        ctrl-A : zooms out to the default "full" view of the plot
        ctrl-+ : moves forward in the zooming stack (if it exists)
        ctrl-- : moves backward in the zooming stack (if it exists)

        """

        ev.accept()
        if ev.text() == '-':
            self.scaleHistory(-1)
        elif ev.text() in ['+', '=']:
            self.scaleHistory(1)
        elif ev.key() == QtCore.Qt.Key_Backspace:
            self.scaleHistory(len(self.axHistory))
        else:
            ev.ignore()
    '''

    '''
    def scaleHistory(self, d):
        if len(self.axHistory) == 0:
            return
        ptr = max(0, min(len(self.axHistory)-1, self.axHistoryPointer+d))
        if ptr != self.axHistoryPointer:
            self.axHistoryPointer = ptr
            self.showAxRect(self.axHistory[ptr])
    '''

    '''
    def updateScaleBox(self, p1, p2):
        r = QtCore.QRectF(p1, p2)
        r = self.getChildGroup().mapRectFromParent(r)
        self.rbScaleBox.resetTransform()
        self.rbScaleBox.setPos(r.topLeft())
        self.rbScaleBox.scale(r.width(), r.height())
        self.rbScaleBox.show()
    '''
    '''
    def showAxRect(self, ax):
        self.setRange(ax.normalized()) # be sure w, h are correct coordinates
        s = self.mouseEnabled()
        self.sigRangeChangedManually.emit(s[0], s[1])
    '''
    '''
    def allChildren(self, item=None):
        """Return a list of all children and grandchildren of this ViewBox"""
        if item is None:
            item = self.getChildGroup()

        children = [item]
        for ch in item.childItems():
            children.extend(self.allChildren(ch))
        return children
    '''
    '''
    def childrenBounds(self, frac=(1.0, 1.0), orthoRange=(None, None), items=tuple()):
        """Return the bounding range of all children.
        [[xmin, xmax], [ymin, ymax]]
        Values may be None if there are no specific bounds for an axis.
        orthoRange is the orthogonal (perpendicular) range
        """
        profiler = debug.Profiler()
        if len(items) == 0:
            items = self.addedItems()

        ## measure pixel dimensions in view box
        px, py = [v.length() if v is not None else 0 for v in self.getChildGroup().pixelVectors()]

        ## First collect all boundary information
        itemBounds = []
        for item in items:
            if not item.isVisible():
                continue

            useX = True
            useY = True

            if hasattr(item, 'dataBounds'):
                if frac is None:
                    frac = (1.0, 1.0)
                xr = item.dataBounds(0, frac=frac[0], orthoRange=orthoRange[0])
                yr = item.dataBounds(1, frac=frac[1], orthoRange=orthoRange[1])
                pxPad = 0 if not hasattr(item, 'pixelPadding') else item.pixelPadding()
                if xr is None or not fn.isfinite(xr[0]) or not fn.isfinite(xr[1]):
                    useX = False
                    xr = (0, 0)
                if yr is None or not fn.isfinite(yr[0]) or not fn.isfinite(yr[1]):
                    useY = False
                    yr = (0, 0)

                bounds = QtCore.QRectF(xr[0], yr[0], xr[1] - xr[0], yr[1] - yr[0])
                bounds = self.mapFromItemToView(item, bounds).boundingRect()

                if not any([useX, useY]):
                    continue

                ## If we are ignoring only one axis, we need to check for rotations
                if useX != useY:  ##   !=  means  xor
                    ang = round(item.transformAngle())
                    if ang == 0 or ang == 180:
                        pass
                    elif ang == 90 or ang == 270:
                        useX, useY = useY, useX
                    else:
                        ## Item is rotated at non-orthogonal angle, ignore bounds entirely.
                        ## Not really sure what is the expected behavior in this case.
                        continue  ## need to check for item rotations and decide how best to apply this boundary.

                itemBounds.append((bounds, useX, useY, pxPad))

            else:
                if int(item.flags() & item.ItemHasNoContents) > 0:
                    continue
                else:
                    bounds = item.boundingRect()
                bounds = self.mapFromItemToView(item, bounds).boundingRect()
                itemBounds.append((bounds, True, True, 0))

        ## determine tentative new range
        rng = [None, None]
        for bounds, useX, useY, px in itemBounds:
            if useY:
                if rng[1] is not None:
                    rng[1] = [min(bounds.top(), rng[1][0]), max(bounds.bottom(), rng[1][1])]
                else:
                    rng[1] = [bounds.top(), bounds.bottom()]
            if useX:
                if rng[0] is not None:
                    rng[0] = [min(bounds.left(), rng[0][0]), max(bounds.right(), rng[0][1])]
                else:
                    rng[0] = [bounds.left(), bounds.right()]
            profiler()

        ## Now expand any bounds that have a pixel margin
        ## This must be done _after_ we have a good estimate of the new range
        ## to ensure that the pixel size is roughly accurate.
        w = self.width()
        h = self.height()
        if w > 0 and rng[0] is not None:
            pxSize = (rng[0][1] - rng[0][0]) / w
            for bounds, useX, useY, px in itemBounds:
                if px == 0 or not useX:
                    continue
                rng[0][0] = min(rng[0][0], bounds.left() - px * pxSize)
                rng[0][1] = max(rng[0][1], bounds.right() + px * pxSize)
        if h > 0 and rng[1] is not None:
            pxSize = (rng[1][1] - rng[1][0]) / h
            for bounds, useX, useY, px in itemBounds:
                if px == 0 or not useY:
                    continue
                rng[1][0] = min(rng[1][0], bounds.top() - px * pxSize)
                rng[1][1] = max(rng[1][1], bounds.bottom() + px * pxSize)

        return rng
    '''

    def childrenBoundingRect(self, frac=Point(1.0, 1.0), orthoRange=(None, None), items=list()):
        rng = self.childrenBounds(frac=frac, orthoRange=orthoRange, items=items)
        tr = self.targetRange()
        if rng[0] is None:
            rng[0] = tr[0]
        if rng[1] is None:
            rng[1] = tr[1]

        bounds = QtCore.QRectF(rng[0][0], rng[1][0], rng[0][1] - rng[0][0], rng[1][1] - rng[1][0])
        return bounds

    '''
    def updateViewRange(self, forceX=False, forceY=False):
        ## Update viewRange to match targetRange as closely as possible, given
        ## aspect ratio constraints. The *force* arguments are used to indicate
        ## which axis (if any) should be unchanged when applying constraints.
        viewRange = self.targetRange()
        changed = [False, False]

        #-------- Make correction for aspect ratio constraint ----------

        # aspect is (widget w/h) / (view range w/h)
        aspect = self.aspectLocked()  # size ratio / view ratio
        tr = self.targetRect()
        bounds = self.rect()
        if aspect != 0.0 and 0 not in [aspect, tr.height(), bounds.height(), bounds.width()]:

            ## This is the view range aspect ratio we have requested
            targetRatio = tr.width() / tr.height() if tr.height() != 0 else 1
            ## This is the view range aspect ratio we need to obey aspect constraint
            viewRatio = (bounds.width() / bounds.height() if bounds.height() != 0 else 1) / aspect
            viewRatio = 1 if viewRatio == 0 else viewRatio

            # Decide which range to keep unchanged
            #print self.name, "aspect:", aspect, "changed:", changed, "auto:", self.state['autoRange']
            if forceX:
                ax = 0
            elif forceY:
                ax = 1
            else:
                # if we are not required to keep a particular axis unchanged,
                # then make the entire target range visible
                ax = 0 if targetRatio > viewRatio else 1

            if ax == 0:
                ## view range needs to be taller than target
                dy = 0.5 * (tr.width() / viewRatio - tr.height())
                if dy != 0:
                    changed[1] = True
                targetRange = self.targetRange()
                viewRange[1] = Range(targetRange[1][0] - dy, targetRange[1][1] + dy)
            else:
                ## view range needs to be wider than target
                dx = 0.5 * (tr.height() * viewRatio - tr.width())
                if dx != 0:
                    changed[0] = True
                targetRange = self.targetRange()
                viewRange[0] = Range(targetRange[0][0] - dx, targetRange[0][1] + dx)


        # ----------- Make corrections for view limits -----------

        #limits = (self.state['limits']['xLimits'], self.state['limits']['yLimits'])
        #minRng = [self.state['limits']['xRange'][0], self.state['limits']['yRange'][0]]
        #maxRng = [self.state['limits']['xRange'][1], self.state['limits']['yRange'][1]]
        limits = (list(self.xLimits()), list(self.yLimits()))
        xRangeLimits = self.xRangeLimits()
        yRangeLimits = self.yRangeLimits()
        minRng = [xRangeLimits[0], yRangeLimits[0]]
        maxRng = [xRangeLimits[1], yRangeLimits[1]]

        for axis in [0, 1]:
            if fn.isnan(limits[axis][0]) and fn.isnan(limits[axis][1]) and fn.isnan(minRng[axis]) and fn.isnan(maxRng[axis]):
                continue

            # max range cannot be larger than bounds, if they are given
            if fn.isfinite(limits[axis][0]) and fn.isfinite(limits[axis][1]):
                if fn.isfinite(maxRng[axis]):
                    maxRng[axis] = min(maxRng[axis], limits[axis][1]-limits[axis][0])
                else:
                    maxRng[axis] = limits[axis][1]-limits[axis][0]

            # Apply xRange, yRange
            diff = viewRange[axis][1] - viewRange[axis][0]
            if fn.isfinite(maxRng[axis]) and diff > maxRng[axis]:
                delta = maxRng[axis] - diff
                changed[axis] = True
            elif fn.isfinite(minRng[axis]) and diff < minRng[axis]:
                delta = minRng[axis] - diff
                changed[axis] = True
            else:
                delta = 0

            viewRange[axis][0] -= delta/2.
            viewRange[axis][1] += delta/2.

            #print "after applying min/max:", viewRange[axis]

            # Apply xLimits, yLimits
            mn, mx = limits[axis]
            if fn.isfinite(mn) and viewRange[axis][0] < mn:
                delta = mn - viewRange[axis][0]
                viewRange[axis][0] += delta
                viewRange[axis][1] += delta
                changed[axis] = True
            elif fn.isfinite(mx) and viewRange[axis][1] > mx:
                delta = mx - viewRange[axis][1]
                viewRange[axis][0] += delta
                viewRange[axis][1] += delta
                changed[axis] = True

            #print "after applying edge limits:", viewRange[axis]

        oldViewRange = self.viewRange()
        changed = [(viewRange[i][0] != oldViewRange[i][0]) or (viewRange[i][1] != oldViewRange[i][1]) for i in (0,1)]
        self.setViewRange(Range(viewRange[0]), Range(viewRange[1]))

        # emit range change signals
        if changed[0]:
            self.sigXRangeChanged.emit(Range(viewRange[0]))
        if changed[1]:
            self.sigYRangeChanged.emit(Range(viewRange[1]))

        if any(changed):
            self.sigRangeChanged.emit(Range(viewRange[0]), Range(viewRange[1]))
            self.update()
            self.setMatrixNeedsUpdate(True)

            # Inform linked views that the range has changed
            for ax in [0, 1]:
                if not changed[ax]:
                    continue
                link = self.linkedView(ax)
                if link is not None:
                    link.linkedViewChanged(self, ax)
    '''

    '''
    def updateMatrix(self, changed=None):
        ## Make the childGroup's transform match the requested viewRange.
        bounds = self.rect()

        vr = self.viewRect()
        if vr.height() == 0 or vr.width() == 0:
            return
        scale = Point(bounds.width()/vr.width(), bounds.height()/vr.height())
        if not self.yInverted():
            scale = scale * Point(1, -1)
        if self.xInverted():
            scale = scale * Point(-1, 1)
        m = QtGui.QTransform()

        ## First center the viewport at 0
        center = bounds.center()
        m.translate(center.x(), center.y())

        ## Now scale and translate properly
        m.scale(scale[0], scale[1])
        st = Point(vr.center())
        m.translate(-st[0], -st[1])

        self.getChildGroup().setTransform(m)

        self.sigTransformChanged.emit()  ## segfaults here: 1
        self.setMatrixNeedsUpdate(False)
    '''
    '''
    def paint(self, p, opt, widget):
        #self.checkSceneChange()

        if self.border is not None:
            bounds = self.shape()
            p.setPen(self.border)
            p.drawPath(bounds)

        #p.setPen(fn.mkPen('r'))
        #path = QtGui.QPainterPath()
        #path.addRect(self.targetRect())
        #tr = self.mapFromView(path)
        #p.drawPath(tr)
    '''

    def updateViewLists(self):
        try:
            self.window()
        except RuntimeError:  ## this view has already been deleted; it will probably be collected shortly.
            return

        def cmpViews(a, b):
            wins = 100 * cmp(a.window() is self.window(), b.window() is self.window())
            alpha = cmp(a.name, b.name)
            return wins + alpha

        ## make a sorted list of all named views
        nv = list(ViewBox.NamedViews.values())
        #print "new view list:", nv
        sortList(nv, cmpViews) ## see pyqtgraph.python2_3.sortList

        if self in nv:
            nv.remove(self)

        ###self.menu.setViewList(nv)

        for ax in [0,1]:
            link = self.linkedView(ax)
            if isinstance(link, basestring):     ## axis has not been linked yet; see if it's possible now
                for v in nv:
                    if link == v.name:
                        self.linkView(ax, v)
        #print "New view list:", nv
        #print "linked views:", self.state['linkedViews']

    @staticmethod
    def updateAllViewLists():
        #print "Update:", ViewBox.AllViews.keys()
        #print "Update:", ViewBox.NamedViews.keys()
        for v in ViewBox.AllViews:
            v.updateViewLists()


    @staticmethod
    def forgetView(vid, name):
        if ViewBox is None:     ## can happen as python is shutting down
            return
        if QtGui.QApplication.instance() is None:
            return
        ## Called with ID and name of view (the view itself is no longer available)
        for v in list(ViewBox.AllViews.keys()):
            if id(v) == vid:
                ViewBox.AllViews.pop(v)
                break
        ViewBox.NamedViews.pop(name, None)
        ViewBox.updateAllViewLists()

    @staticmethod
    def quit():
        ## called when the application is about to exit.
        ## this disables all callbacks, which might otherwise generate errors if invoked during exit.
        for k in ViewBox.AllViews:
            if isQObjectAlive(k) and getConfigOption('crashWarning'):
                sys.stderr.write('Warning: ViewBox should be closed before application exit.\n')

            try:
                k.destroyed.disconnect()
            except RuntimeError:  ## signal is already disconnected.
                pass
            except TypeError:  ## view has already been deleted (?)
                pass
            except AttributeError:  # PySide has deleted signal
                pass

    def locate(self, item, timeout=3.0, children=False):
        """
        Temporarily display the bounding rect of an item and lines connecting to the center of the view.
        This is useful for determining the location of items that may be out of the range of the ViewBox.
        if allChildren is True, then the bounding rect of all item's children will be shown instead.
        """
        self.clearLocate()

        if item.scene() is not self.scene():
            raise Exception("Item does not share a scene with this ViewBox.")

        c = self.viewRect().center()
        if children:
            br = self.mapFromItemToView(item, item.childrenBoundingRect()).boundingRect()
        else:
            br = self.mapFromItemToView(item, item.boundingRect()).boundingRect()

        g = ItemGroup()
        g.setParentItem(self.getChildGroup())
        self.locateGroup = g
        g.box = QtGui.QGraphicsRectItem(br)
        g.box.setParentItem(g)
        g.lines = []
        for p in (br.topLeft(), br.bottomLeft(), br.bottomRight(), br.topRight()):
            line = QtGui.QGraphicsLineItem(c.x(), c.y(), p.x(), p.y())
            line.setParentItem(g)
            g.lines.append(line)

        for item in g.childItems():
            item.setPen(fn.mkPen(color='y', width=3))
        g.setZValue(1000000)

        if children:
            g.path = QtGui.QGraphicsPathItem(g.childrenShape())
        else:
            g.path = QtGui.QGraphicsPathItem(g.shape())
        g.path.setParentItem(g)
        g.path.setPen(fn.mkPen('g'))
        g.path.setZValue(100)

        QtCore.QTimer.singleShot(timeout*1000, self.clearLocate)

    def clearLocate(self):
        if self.locateGroup is None:
            return
        self.scene().removeItem(self.locateGroup)
        self.locateGroup = None

from .ViewBoxMenu import ViewBoxMenu
