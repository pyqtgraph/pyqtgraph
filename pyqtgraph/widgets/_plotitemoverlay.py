from ..Qt.QtWidgets import QGraphicsGridLayout
from ..functions import connect_lambda
# from ..graphicsItems.AxisItem import AxisItem
# from ..graphicsItems.GraphicsObject import GraphicsObject
from ..graphicsItems.PlotItem.PlotItem import PlotItem
from ..graphicsItems.ViewBox import ViewBox
from ..Qt.QtCore import QObject, Signal, Qt

# Bugs TODO:
# - figure out why per-axis drag clicking is halted
#   after a single handler call - seems like it's a more serious
#   bug in the depths of the viewbox handlers rat's nest XD

# Unimplemented features TODO:
# - 'A' (autobtn) should relay to all views
# - layout unwind and re-pack for 'left' and 'top' axes
# - add labels to layout if detected in source ``PlotItem``

# UX nice-to-have TODO:
# - optional "focussed" view box support for view boxes
#   that have custom input handlers (eg. you might want to
#   scale the view to some "focussed" data view and have overlayed
#   viewboxes only respond to relayed events.)
# - figure out how to deal with menu raise events for multi-viewboxes.
#   (we might want to add a different menu which specs the name of the
#   view box currently being handled?
# - allow selection of a particular view box by interacting with its
#   axis?

__all__ = ["PlotItemOverlay"]

# Define the layout "position" indices as to be passed
# to a ``QtWidgets.QGraphicsGridlayout.addItem()`` call:
# https://doc.qt.io/qt-5/qgraphicsgridlayout.html#addItem
# This was pulled from the internals of ``PlotItem.setAxisItem()``.
_axes_layout_indices: dict[str] = {
    # row incremented axes
    'top': (1, 1),
    'bottom': (3, 1),

    # column incremented axes
    'left': (2, 0),
    'right': (2, 2),
}
# NOTE: To clarify this indexing, ``PlotItem.__init__()`` makes a grid
# with dimensions 4x3 and puts the ``ViewBox`` at postiion (2, 1) (aka
# row=2, col=1) in the grid layout since row (0, 1) is reserved for
# a title label and row 1 is for any potential "top" axis. Column 1
# is the "middle" (since 3 columns) and is where the plot/vb is placed.


class PlotItemOverlay:
    '''
    A composite for managing overlaid ``PlotItem`` instances such that
    you can make multiple graphics appear on the same graph with
    separate (non-colliding) axes apply ``ViewBox`` signal broadcasting
    such that all overlaid items respond to input simultaneously.

    '''
    def __init__(
        self,
        root_plotitem: PlotItem
    ) -> None:
        self.root_plotitem: PlotItem = root_plotitem
        root_plotitem.vb.allow_signal_relay
        self.overlays: list[PlotItem] = []
        self._relays: dict[str, Signal] = {}

    @property
    def layout(self) -> QGraphicsGridLayout:
        '''
        Return reference to the "focussed" ``PlotItem``'s grid layout.

        '''
        return self.root_plotitem.layout

    def add_plotitem(
        self,
        plotitem: PlotItem,

        # TODO: we could also put the ``ViewBox.XAxis``
        # style enum here?
        link_axes: tuple[int] = (),

        # XXX: all the other optional inputs.
        # should we enum or at least type check this?
        # link_axes: tuple[int] = (0,),  # link x
        # link_axes: tuple[int] = (1,),  # link y
        # link_axes: tuple[int] = (0, 1),  # link both

    ) -> None:

        root = self.root_plotitem
        layout: QGraphicsGridLayout = root.layout
        self.overlays.append(plotitem)

        vb: ViewBox = plotitem.vb

        # can't do this since only root will get
        # menu event then..
        # TODO: some sane way to allow menu event broadcast XD
        # vb.setMenuEnabled(False)

        # instruct all vbs to relay signals instead
        # of consuming without relay
        vb.allow_signal_relay = True

        # TODO: move this into ``ViewBox`` as some kind of special
        # linking method for full view box event relaying.
        if not link_axes:
            # TODO: there's still an issue with manually click-dragging
            # axes - not sure what that's about but likely some bug
            # inside the mangled mess of event handling that is the
            # `ViewBox` core XD

            # NOTE: for this to work the ``.allow_signal_relay`` patch
            # made to ``ViewBox`` needs to exist to avoid events being
            # consumed too early.
            self._relays[plotitem] = root.vb.sigMouseDragged.connect(
                vb.mouseDragEvent)
            self._relays[plotitem] = root.vb.sigMouseWheel.connect(vb.wheelEvent)

        else:
            for dim in link_axes:
                # link x and y axes to new view box such that the top level
                # viewbox propagates to the root (and whatever other plotitem
                # overlays that have been added).
                # vb.linkView(dim, root.vb)
                root.vb.linkView(dim, vb)

        # make overlaid viewbox impossible to focus since the top
        # level should handle all input and relay to overlays.
        # TODO: we will probably want to add a "focus" api such that
        # a new "top level" ``PlotItem`` can be selected dynamically
        # (and presumably the axes dynamically sorted to match).
        vb.setFlag(
            vb.GraphicsItemFlag.ItemIsFocusable,
            False
        )
        vb.setFocusPolicy(Qt.NoFocus)

        # Add any axes in appropriate sequence to the top level layout
        # to avoid graphics collision.
        count = len(self.overlays)
        assert count

        for name, axis_info in plotitem.axes.items():
            axis = axis_info['item']
            axis_view = axis.linkedView()

            # Remove old axis
            # plotitem.removeAxis(axis, unlink=False)
            if not axis.isVisible():
                continue

            plotitem.removeAxis(axis, unlink=False)

            # vb.linkView(0, "")
            # vb.linkView(1, "")
            # scene = axis.scene()
            # if scene:
            #     scene.removeItem(axis)

            # XXX: DON'T unlink it since we the original ``ViewBox``
            # to still drive it B)
            # axis.unlinkFromView()

            # if not axis.isVisible():
            # if name != 'right':
            #     continue

            if name in ('top', 'bottom'):
                i_dim = 0
            elif name in ('left', 'right'):
                i_dim = 1

            # TODO: increment logic for layout on 'top'/'left' axes
            # sets.. looks like ther'es no way around an unwind and
            # re-stack of the layout to include all labels, unless
            # we use a different layout system (cough).

            # if name in ('top', 'left'):
            #     increment = -1
            # elif name in ('right', 'bottom'):
            #     increment = +1

            increment = +count

            index = list(_axes_layout_indices[name])
            current = index[i_dim]
            index[i_dim] = current + increment if current > 0 else 0

            # layout re-stack logic avoid collision
            # with existing indices.
            item = layout.itemAt(*index)
            while item:
                index[i_dim] += 1
                item = layout.itemAt(*index)

            layout.addItem(axis, *index)

        # overlay plot item's view with parent
        # yes, y'all were right we do need this B)
        plotitem.setGeometry(self.root_plotitem.vb.sceneBoundingRect())
        connect_lambda(
            root.vb.sigResized,
            plotitem,
            lambda plotitem,
            vb: plotitem.setGeometry(vb.sceneBoundingRect())
        )
        # ensure the overlayed view is redrawn on each cycle
        root.scene().sigPrepareForPaint.connect(vb.prepareForPaint)

        # vb.sigStateChanged.emit(vb)

        # focus state sanity
        vb.clearFocus()
        assert not vb.focusWidget()
        root.vb.setFocus()
        assert root.vb.focusWidget()

    # XXX: do we need this? Why would you build then destroy?
    def remove_plotitem(self, plotItem: PlotItem) -> None:
        '''
        Remove this ``PlotItem`` from the overlayed set making not shown
        and unable to accept input.

        '''
        ...

    # TODO: i think this would be super hot B)
    def focus_item(self, plotitem: PlotItem) -> PlotItem:
        '''
        Apply focus to a contained PlotItem thus making it the "top level"
        item in the overlay able to accept peripheral's input from the user
        and responsible for zoom and panning control via its ``ViewBox``.

        '''
        ...

    # TODO: i guess we need this if you want to detach existing plots
    # dynamically? XXX: untested as of now.
    def _disconnect_all(
        self,
        plotitem: PlotItem,
    ) -> list[Signal]:
        '''
        Disconnects all signals related to this widget for the given chart.

        '''
        disconnected = []
        for pi, sig in self._relays.items():
            QObject.disconnect(sig)
            disconnected.append(sig)

        return disconnected
