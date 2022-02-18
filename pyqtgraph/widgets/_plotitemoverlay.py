# import inspect
from typing import Callable, Optional

from ..Qt.QtWidgets import QGraphicsGridLayout
from ..graphicsItems.AxisItem import AxisItem
from ..graphicsItems.ViewBox import ViewBox
from ..graphicsItems.GraphicsWidget import GraphicsWidget
from ..graphicsItems.PlotItem.PlotItem import PlotItem
from ..Qt.QtCore import QObject, Signal, Qt, QEvent

# Unimplemented features TODO:
# - 'A' (autobtn) should relay to all views
# - context menu single handler + relay?
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


# TODO: we might want to enabled some kind of manual flag to disable
# this method wrapping during type creation? As example a user could
# definitively decide **not** to enable broadcasting support by
# setting something like ``ViewBox.disable_relays = True``?
def mk_relay_method(

    signame: str,
    slot: Callable[
        [ViewBox,
         'QEvent',
         Optional[AxisItem]],
        None,
    ],

) -> Callable[
    [
        ViewBox,
        # lol, there isn't really a generic type thanks
        # to the rewrite of Qt's event system XD
        'QEvent',

        'Optional[AxisItem]',
        'Optional[ViewBox]',  # the ``relayed_from`` arg we provide
    ],
    None,
]:

    def maybe_broadcast(
        vb: 'ViewBox',
        ev: 'QEvent',
        axis: 'Optional[int]' = None,
        relayed_from: 'ViewBox' = None,

    ) -> None:
        '''
        (soon to be) Decorator which makes an event handler
        "broadcastable" to overlayed ``GraphicsWidget``s.

        Adds relay signals based on the decorated handler's name
        and conducts a signal broadcast of the relay signal if there
        are consumers registered.

        '''
        # When no relay source has been set just bypass all
        # the broadcast machinery.
        if vb.event_relay_source is None:
            ev.accept()
            return slot(
                vb,
                ev,
                axis=axis,
            )

        if relayed_from:
            assert axis is None

            # this is a relayed event and should be ignored (so it does not
            # halt/short circuit the graphicscene loop). Further the
            # surrounding handler for this signal must be allowed to execute
            # and get processed by **this consumer**.
            print(f'{vb.name} rx relayed from {relayed_from.name}')
            ev.ignore()

            return slot(
                vb,
                ev,
                axis=axis,
            )

        if axis is not None:
            print(f'{vb.name} handling axis event:\n{str(ev)}')
            ev.accept()
            return slot(
                vb,
                ev,
                axis=axis,
            )

        elif (
            relayed_from is None
            and vb.event_relay_source is vb  # we are the broadcaster
            and axis is None
        ):
            # Broadcast case: this is a source event which will be
            # relayed to attached consumers and accepted after all
            # consumers complete their own handling followed by this
            # routine's processing. Sequence is,
            # - pre-relay to all consumers *first* - ``.emit()`` blocks
            #   until all downstream relay handlers have run.
            # - run the source handler for **this** event and accept
            #   the event

            # Access the "bound signal" that is created
            # on the widget type as part of instantiation.
            signal = getattr(vb, signame)
            print(f'{vb.name} emitting {signame}')

            # TODO/NOTE: we could also just bypass a "relay" signal
            # entirely and instead call the handlers manually in
            # a loop? This probably is a lot simpler and also doesn't
            # have any downside, and allows not touching target widget
            # internals.
            signal.emit(
                ev,
                axis,
                # passing this demarks a broadcasted/relayed event
                vb,
            )
            # accept event so no more relays are fired.
            ev.accept()

            # call underlying wrapped method with an extra ``relayed_from`` value
            # to denote that this is a relayed event handling case.
            return slot(
                vb,
                ev,
                axis=axis,
            )

    return maybe_broadcast


# XXX: :( can't define signals **after** class compile time
# so this is not really useful.
# def mk_relay_signal(
#     func,
#     name: str = None,

# ) -> Signal:
#     (
#         args,
#         varargs,
#         varkw,
#         defaults,
#         kwonlyargs,
#         kwonlydefaults,
#         annotations
#     ) = inspect.getfullargspec(func)

#     # XXX: generate a relay signal with 1 extra
#     # argument for a ``relayed_from`` kwarg. Since
#     # ``'self'`` is already ignored by signals we just need
#     # to count the arguments since we're adding only 1 (and
#     # ``args`` will capture that).
#     numargs = len(args + list(defaults))
#     signal = Signal(*tuple(numargs * [object]))
#     signame = name or func.__name__ + 'Relay'
#     return signame, signal


def enable_relays(
    widget: GraphicsWidget,
    handler_names: list[str],

) -> list[Signal]:
    '''
    Method override helper which enables relay of a particular
    ``Signal`` from some chosen broadcaster widget to a set of
    consumer widgets which should operate their event handlers normally
    but instead of signals "relayed" from the broadcaster.

    Mostly useful for overlaying widgets that handle user input
    that you want to overlay graphically. The target ``widget`` type must
    define ``QtCore.Signal``s each with a `'Relay'` suffix for each
    name provided in ``handler_names: list[str]``.

    '''
    signals = []
    for name in handler_names:
        handler = getattr(widget, name)
        signame = name + 'Relay'
        # ensure the target widget defines a relay signal
        relay = getattr(widget, signame)
        widget.relays[signame] = name
        signals.append(relay)
        method = mk_relay_method(signame, handler)
        setattr(widget, name, method)

    return signals


enable_relays(
    ViewBox,
    ['wheelEvent', 'mouseDragEvent']
)


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

        # method = mk_relay_method(relay_signal_name, handler)
        vb = root_plotitem.vb
        vb.event_relay_source = vb  # TODO: maybe change name?
        vb.setZValue(1000)  # XXX: critical for scene layering/relaying

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
        # (0,),  # link x
        # (1,),  # link y
        # (0, 1),  # link both
        link_axes: tuple[int] = (),

    ) -> None:

        root = self.root_plotitem
        layout: QGraphicsGridLayout = root.layout
        self.overlays.append(plotitem)
        vb: ViewBox = plotitem.vb

        # mark this consumer overlay as ready to expect relayed events
        # from the root plotitem.
        vb.event_relay_source = root.vb

        # TODO: some sane way to allow menu event broadcast XD
        # vb.setMenuEnabled(False)

        # TODO: inside the `maybe_broadcast()` (soon to be) decorator
        # we need have checks that consumers have been attached to
        # these relay signals.
        if link_axes != (0, 1):

            # wire up relay signals
            for relay_signal_name, handler_name in vb.relays.items():
                # print(handler_name)
                # XXX: Signal class attrs are bound after instantiation
                # of the defining type, so we need to access that bound
                # version here.
                signal = getattr(root.vb, relay_signal_name)
                handler = getattr(vb, handler_name)
                # slot = mk_relay_method(relay_signal_name, handler)
                # breakpoint()
                signal.connect(handler)

                # sig = root.vb.sigMouseDraggedRelay.connect(
                #     partial(
                #         vb.mouseDragEvent,
                #     )

                # )

        # link dim-axes to root if requested by user.
        # TODO: solve more-then-wanted scaled panning on click drag
        # which seems to be due to broadcast. So we probably need to
        # disable broadcast when axes are linked in a particular
        # dimension?
        for dim in link_axes:
            # link x and y axes to new view box such that the top level
            # viewbox propagates to the root (and whatever other
            # plotitem overlays that have been added).
            vb.linkView(dim, root.vb)

        # make overlaid viewbox impossible to focus since the top
        # level should handle all input and relay to overlays.
        # NOTE: this was solved with the `setZValue()` above!

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
            # axis_view = axis.linkedView()

            # Remove old axis
            # plotitem.removeAxis(axis, unlink=False)
            if not axis.isVisible():
                continue

            # XXX: DON'T unlink it since we the original ``ViewBox``
            # to still drive it B)
            plotitem.removeAxis(axis, unlink=False)

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

        plotitem.setGeometry(root.vb.sceneBoundingRect())

        def size_to_viewbox(vb: 'ViewBox'):
            plotitem.setGeometry(vb.sceneBoundingRect())

        root.vb.sigResized.connect(size_to_viewbox)

        # ensure the overlayed view is redrawn on each cycle
        root.scene().sigPrepareForPaint.connect(vb.prepareForPaint)

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
