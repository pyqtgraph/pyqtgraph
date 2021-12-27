from ..Qt.QtWidgets import QGraphicsGridLayout
from ..functions import connect_lambda
# from ..graphicsItems.AxisItem import AxisItem
# from ..graphicsItems.GraphicsObject import GraphicsObject
# from ..graphicsItems.PlotDataItem import PlotDataItem
from ..graphicsItems.PlotItem.PlotItem import PlotItem
from ..graphicsItems.ViewBox import ViewBox
from ..Qt.QtCore import QObject, Signal


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
        self.overlays: list[PlotItem] = []
        self._signal_relay: dict[str, Signal] = {}

    @property
    def layout(self) -> QGraphicsGridLayout:
        '''
        Return reference to the "focussed" ``PlotItem``'s grid layout.

        '''
        return self.root_plotitem.layout

    # Add/Remove API which allows for dynamic mutation
    # of the overlayed ``PlotItem`` collection.
    def add_plotitem(
        self,
        plotitem: PlotItem,

        # TODO: we could also put the ``ViewBox.XAxis``
        # style enum here?
        link_axes: tuple[int] = (0,),

    ) -> None:

        root = self.root_plotitem
        layout: QGraphicsGridLayout = root.layout
        self.overlays.append(plotitem)

        vb: ViewBox = plotitem.vb
        for dim in link_axes:
            # link x and y axes to new view box such that the top level
            # viewbox propagates to the root (and whatever other plotitem
            # overlays that have been added).
            vb.linkView(dim, root.vb)

        # XXX: when would you ever want the y-axis for overlaid plots to
        # be linked? Seems like nonsense presuming the whole point of
        # overlays is disjoint co-domains?
        # vb.linkView(1, root.vb)

        # make overlaid viewbox impossible to focus since the top
        # level should handle all input and relay to overlays.
        # TODO: we will probably want to add a "focus" api such that
        # a new "top level" ``PlotItem`` can be selected dynamically
        # (and presumably the axes dynamically sorted to match).
        vb.setFlag(vb.GraphicsItemFlag.ItemIsFocusable, False)

        # breakpoint()
        root.vb.setFocus()

        # Add any axes in appropriate sequence to the top level layout
        # to avoid graphics collision.
        count = len(self.overlays)
        assert count
        try:
            for name, axis_info in plotitem.axes.items():
                axis = axis_info['item']

                # plotitem.hideAxis(name)

                # Remove old axis
                plotitem.layout.removeItem(axis)
                axis.scene().removeItem(axis)

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

                # breakpoint()
                index = list(_axes_layout_indices[name])
                index[i_dim] = index[i_dim] + count
                # breakpoint()
                out = layout.addItem(axis, *index)
                if out:
                    breakpoint()

        except Exception as _err:
            err = _err
            breakpoint()

        # overlay plot item's view with parent
        plotitem.setGeometry(self.root_plotitem.vb.sceneBoundingRect())
        connect_lambda(
            root.vb.sigResized,
            plotitem,
            lambda plotitem,
            vb: plotitem.setGeometry(vb.sceneBoundingRect())
        )

        # TODO: move this into ``ViewBox`` as some kind of special
        # linking method for full view box event relaying.
        # if link_all_vbs:  # or wtv
        #     # FROM "https://github.com/pyqtgraph/pyqtgraph/pull/2010" by
        #     # herodotus77 propagate mouse actions to charts "hidden"
        #     # behind
        #     root.vb.sigMouseDragged.connect(vb.mouseDragEvent)
        #     root.vb.sigMouseWheel.connect(vb.wheelEvent)
        #     root.vb.sigHistoryChanged.connect(vb.scaleHistory)

    def remove_plotitem(self, plotItem: PlotItem) -> None:
        '''
        Remove this ``PlotItem`` from the overlayed set making not shown
        and unable to accept input.

        '''
        ...

    def focus_item(self, plotitem: PlotItem) -> PlotItem:
        '''
        Apply focus to a contained PlotItem thus making it the "top level"
        item in the overlay able to accept peripheral's input from the user
        and responsible for zoom and panning control via its ``ViewBox``.

        '''
        ...

    def _disconnect_all(self, chart) -> None:
        '''
        Disconnects all signals related to this widget for the given chart.

        '''
        signals = self._signalConnectionsByChart[chart.name]
        for conn_name, conn in signals.items():
            if conn is not None:
                QObject.disconnect(conn)
                signals[conn_name] = None
