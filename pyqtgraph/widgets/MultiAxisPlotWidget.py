__all__ = ["MultiAxisPlotWidget"]

import weakref
from collections import namedtuple
from typing import Dict, List

from ..functions import disconnect, prep_lambda_for_connect
from ..graphicsItems.AxisItem import AxisItem
from ..graphicsItems.PlotDataItem import PlotDataItem
from ..graphicsItems.PlotItem.PlotItem import PlotItem
from ..graphicsItems.ViewBox import ViewBox
from ..widgets.PlotWidget import PlotWidget

SigDisconnector = namedtuple("SigDisconnector", ["signal", "slot", "meta"])


class MultiAxisPlotWidget(PlotWidget):
    def __init__(self, **kwargs):
        """PlotWidget but with support for multi axis.
        usage: you generally want to call in this order:
        .addAxis(...) as many times as axes needed,
        .addChart(...) as many times as charts needed.
        Also consider calling .update() after updating the chart data.
        Refer to the example named :class:MultiAxisPlotWidgetExample."""
        super().__init__(enableMenu=False, **kwargs)
        # plotitem shortcut
        self.pi = super().getPlotItem()
        # override autorange button behaviour
        self.pi.autoBtn.clicked.disconnect()
        self.pi.autoBtn.clicked.connect(
            prep_lambda_for_connect(self, lambda widg, button: (widg.enableAxisAutoRange(), widg.update()))
        )
        # default vb from plotItem shortcut
        self.vb = self.pi.vb
        # layout shortcut
        self.layout = self.pi.layout
        # CHARTS
        self.axes: List[AxisItem] = []
        # select custom behaviour
        self.pi.extra_axes = self.axes
        self.charts: List[PlotDataItem] = []
        self.plot_items: Dict[PlotDataItem, PlotItem] = {}
        self._signalConnectionsByChart: Dict[PlotDataItem, Dict] = {}
        self._signalConnectionsByAxis: Dict[AxisItem, Dict] = {}

    def addAxis(self, *args, axis=None, **kwargs):
        """Add a new axis (AxisItem) to the widget, will be shown and linked to a
        chart when used in addChart().

        Parameters
        ----------
        axis : AxisItem, optional
            The axis to be used. If left as None a new AxisItem will be created from args and kwargs.
        args : iterable, optional
            arguments to be passed to the newly created AxisItem
        kwargs : dict, optional
            arguments to be passed to the newly created AxisItem.  Remember to
            set the AxisItem orientation.

        Returns
        -------
        AxisItem
        """
        if axis in self.axes:
            raise ValueError("Axis already added")
        if axis is None:
            axis = AxisItem(*args, **kwargs)
        axis.autorange = False
        axis.charts = []
        self.axes.append(axis)
        if axis not in self._signalConnectionsByAxis:
            self._signalConnectionsByAxis[axis] = {}
        axis.showLabel(True)
        self.makeLayout(axes=self.axes, charts=self.charts)
        return axis

    def addChart(self, xAxis=None, yAxis=None, chart=None, *args, **kwargs):
        """Add a new chart (PlotDataItem in a PlotItem) to the widget, will be
        shown when used in makeLayout().

        Parameters
        ----------
        xAxis : Union[AxisItem, str], optional
            one of the default PlotItem axis names ("top", "right", "bottom",
            "left"), or any axis previously added by calling ``addAxis``. If
            omitted "bottom" will be used as default.
        yAxis : Union[AxisItem, str], optional
            one of the default PlotItem axis names ("top", "right", "bottom",
            "left"), or any axis previously added by calling ``addAxis``. If
            omitted "left" will be used as default.
        chart : PlotDataItem, optional
            The chart to be used inside the new PlotItem. If left as None a
            new PlotDataItem will be created.
        args : iterable, optional
            Arguments to be passed to the newly created PlotItem. See
            :class:`~pyqtgraph.PlotItem`
        kwargs : dict, optional
            Parameters to be passed to the newly created PlotItem. See
            :class:`~pyqtgraph.PlotItem`

        Returns
        -------
        PlotDataItem
            The newly created PlotDataItem.
        PlotItem
            The newly created PlotItem.
        """
        if chart in self.charts:
            raise ValueError("Chart has already been added")
        if xAxis is None:
            xAxis = "bottom"
        if isinstance(xAxis, str):
            # add default axis to the list of axes if requested
            xAxis = self.addAxis(orientation=xAxis)  # , axis=self.pi.axes[xAxis]["item"])  # TODO
        elif xAxis not in self.axes:
            self.addAxis(xAxis)
        if yAxis is None:
            yAxis = "left"
        if isinstance(yAxis, str):
            # add default axis to the list of axes if requested
            yAxis = self.addAxis(orientation=yAxis)  # , axis=self.pi.axes[yAxis]["item"])  # TODO
        elif yAxis not in self.axes:
            self.addAxis(yAxis)
        if xAxis in {"bottom", "top"} and yAxis in {"left", "right"}:
            # use default plotitem if none provided
            plotitem = self.pi
        else:
            # VIEW
            plotitem = PlotItem(parent=self.pi, *args, enableMenu=False, **kwargs,)
            # disable buttons (autoranging)
            plotitem.hideButtons()
            # fix parent legend not showing child charts
            plotitem.legend = self.pi.legend
            layouts = {
                "left": [2, 0],
                "bottom": [3, 1],
                "right": [2, 2],
                "top": [1, 1],
            }

            # hide all existing axes:
            for orientation in layouts:
                plotitem.hideAxis(orientation)

            plotitem.axes[xAxis.orientation] = {"item": xAxis, "pos": layouts[xAxis.orientation]}
            plotitem.axes[yAxis.orientation] = {"item": yAxis, "pos": layouts[yAxis.orientation]}
        # CHART
        if chart is None:
            chart = PlotDataItem()
        plotitem.addItem(chart)
        # keep plotitem inside chart
        self.plot_items[chart] = plotitem
        # keep track of connections
        if chart not in self._signalConnectionsByChart:
            self._signalConnectionsByChart[chart] = {}
        # keep axis track
        chart.axes = [xAxis, yAxis]
        # keep chart
        self.charts.append(chart)
        # create a mapping for this chart and his axis
        xAxis.charts.append(chart)
        yAxis.charts.append(chart)
        self.makeLayout(axes=self.axes, charts=self.charts)
        return chart, plotitem

    def clearLayout(self):
        """Clears the widget but keeps all created axis and charts."""
        while self.layout.count() > 0:
            item = self.layout.itemAt(0)
            self.layout.removeAt(0)
            self.scene().removeItem(item)
        for chart in self.charts:
            self._chart_disconnect_all(chart)
        for axis in self.axes:
            self._axis_disconnect_all(axis)

    def makeLayout(self, axes=None, charts=None):
        """Adds all given axes and charts to the widget.

        Parameters
        ----------
        axes : List[AxisItem], optional
            The names associated with the axes to show.
            Axes will be ordered from left-to-right and from top-to-bottom
            following the given axes list order if given,
            or the axis creation order otherwise.
        charts : List[PlotDataItems], optional
            The names associated with the charts to show.
        """
        self.clearLayout()
        self._show_axes(axes)
        self._show_charts(charts)
        for chart in self.charts:
            self._connect_signals(chart)
        self.update()

    def _show_axes(self, axes=None):
        """Shows all the selected axes."""
        if axes is None:
            axes = self.axes
        axes = list(dict.fromkeys(axes))
        lo = {
            "left": [],
            "right": [],
            "top": [],
            "bottom": [],
        }
        for axis in axes:
            # add margin to the axis label to avoid overlap of other axis ticks
            axis.setStyle(labelMargin=[0, 0])
            lo[axis.orientation].append(axis)
        horizontal_axes_pos_x = len(lo["left"])
        vertical_axes_pos_y = 1 + len(lo["top"])
        # ADD TITLE ON TOP
        self.pi.titleLabel.show()  # TODO associate this with the toplevel layout, not a plotitem
        self.layout.addItem(self.pi.titleLabel, 0, horizontal_axes_pos_x)
        # ADD MAIN PLOTITEM
        self.vb.show()
        self.layout.addItem(self.vb, vertical_axes_pos_y, horizontal_axes_pos_x)
        # ADD AXIS
        for axes_pos_x, axis in enumerate(lo["left"]):
            axis.show()
            self.layout.addItem(axis, vertical_axes_pos_y, axes_pos_x)
        for axes_pos_x, axis in enumerate(lo["right"], start=len(lo["left"]) + 2):
            axis.show()
            self.layout.addItem(axis, vertical_axes_pos_y, axes_pos_x)
        for axes_pos_y, axis in enumerate(lo["top"], start=1):
            axis.show()
            self.layout.addItem(axis, axes_pos_y, horizontal_axes_pos_x)
        for axes_pos_y, axis in enumerate(lo["bottom"], start=len(lo["top"]) + 3):
            axis.show()
            self.layout.addItem(axis, axes_pos_y, horizontal_axes_pos_x)

    def _show_charts(self, charts=None):
        """Shows all (and only) the selected charts.

        charts : List[PlotDataItem], optional
        """
        if charts is None:
            charts = self.charts
        for chart in set(self.charts) - set(charts):
            chart.hide()
        for chart in charts:
            chart.show()

    def _connect_signals(self, chart: PlotDataItem):
        """Connects all signals related to this widget for the given chart given the top level one."""
        chart_vb = self.plot_items[chart].vb
        signals = self._signalConnectionsByChart[chart]
        scene = self.scene()

        def connectify(holder, name, signal, slot):
            meta = signal.connect(slot)
            holder[name] = SigDisconnector(signal=signal, slot=slot, meta=meta)

        for axis in chart.axes:
            # link axis to view
            # connect view changes to axis
            # FROM AxisItem.linkToView
            # connect view changes to axis changes
            if axis.orientation in {"top", "bottom"}:
                connectify(signals, "propagate chart range to axis", chart_vb.sigXRangeChanged, axis.linkedViewChanged)
            elif axis.orientation in {"right", "left"}:
                connectify(signals, "propagate chart range to axis", chart_vb.sigYRangeChanged, axis.linkedViewChanged)
            connectify(signals, "propagate chart resize to axis", chart_vb.sigResized, axis.linkedViewChanged)
            # set axis main view link if not assigned
            if axis.linkedView() is None:
                axis._linkedView = weakref.ref(chart_vb)
                axis_vb = chart_vb
            else:
                axis_vb = axis.linkedView()
            # FROM ViewBox.linkView
            # connect axis's view changes to view since axis acts just like a proxy to it
            if axis.orientation in {"top", "bottom"}:
                # connect axis main view changes to view
                chart_vb.state["linkedViews"][chart_vb.XAxis] = weakref.ref(axis_vb)
                # this signal is received multiple times when using mouse actions directly on the viewbox
                # this causes the non-top layer views to scroll more than the front-most one
                connectify(signals, "propagate axis range to chart", axis_vb.sigXRangeChanged, chart_vb.linkedXChanged)
                connectify(signals, "propagate axis resize to chart", axis_vb.sigResized, chart_vb.linkedXChanged)
            elif axis.orientation in {"right", "left"}:
                # connect axis main view changes to view
                chart_vb.state["linkedViews"][chart_vb.YAxis] = weakref.ref(axis_vb)
                # this signal is received multiple times when using mouse actions directly on the viewbox
                # this causes the non-top layer views to scroll more than the front-most one
                connectify(signals, "propagate axis range to chart", axis_vb.sigYRangeChanged, chart_vb.linkedYChanged)
                connectify(signals, "propagate axis resize to chart", axis_vb.sigResized, chart_vb.linkedYChanged)
            axis_signals = self._signalConnectionsByAxis[axis]
            if "disable axis autorange on axis manual change" not in axis_signals:
                if axis.orientation in {"top", "bottom"}:
                    # disable autorange on manual movements
                    connectify(
                        axis_signals,
                        "disable axis autorange on axis manual change",
                        axis_vb.sigXRangeChangedManually,
                        prep_lambda_for_connect(self, lambda widg, mask: widg.disableAxisAutoRange(axis)),
                    )
                elif axis.orientation in {"right", "left"}:
                    # disable autorange on manual movements
                    connectify(
                        axis_signals,
                        "disable axis autorange on axis manual change",
                        axis_vb.sigYRangeChangedManually,
                        prep_lambda_for_connect(self, lambda widg, mask: widg.disableAxisAutoRange(axis)),
                    )
            chart_vb.sigStateChanged.emit(chart_vb)
        # resize plotitem according to the master one
        # resizing its view doesn't work for some reason
        if chart_vb is not self.vb:
            # make default vb always the top level one chart vb
            chart_vb.setZValue(0)
            self.vb.setZValue(9999)  # over 9thousand!
            chart_pi = self.plot_items[chart]
            # using connect_lambda here as a workaround
            # refer to the documentation of connect_lambda in functions.py for an explanation of the issue
            connectify(
                signals,
                "propagate default vb resize to chart",
                self.vb.sigResized,
                prep_lambda_for_connect(chart_pi, lambda _pi, vb: _pi.setGeometry(vb.sceneBoundingRect())),
            )
            # FROM "https://github.com/pyqtgraph/pyqtgraph/pull/2010" by herodotus77
            # propagate mouse actions to the charts "hidden" behind
            connectify(
                signals,
                "propagate top level mouse drag interaction to chart",
                self.vb.sigMouseDragged,
                chart_vb.mouseDragEvent,
            )
            connectify(
                signals,
                "propagate top level mouse wheel interaction to chart",
                self.vb.sigMouseWheel,
                chart_vb.wheelEvent,
            )
            connectify(
                signals,
                "propagate top level history changes interaction to chart",
                self.vb.sigHistoryChanged,
                chart_vb.scaleHistory,
            )
            connectify(
                signals,
                "propagate default scene prepare_for_paint to chart",
                scene.sigPrepareForPaint,
                chart_vb.prepareForPaint,
            )

    def _chart_disconnect_all(self, chart):
        """Disconnects all signals related to this widget for the given chart."""
        signals = self._signalConnectionsByChart[chart]
        for conn_name in list(signals.keys()):
            sig, slot, meta = signals.pop(conn_name)
            disconnect(sig, slot, meta)

    def _axis_disconnect_all(self, axis):
        """Disconnects all signals related to this widget for the given axis."""
        signals = self._signalConnectionsByAxis[axis]
        for conn_name in list(signals.keys()):
            sig, slot, meta = signals.pop(conn_name)
            disconnect(sig, slot, meta)

    def clean(self):
        """Clears all charts' contents."""
        # CLEAR PLOTS
        for chart in self.charts:
            chart.clear()
        self.update()

    def getPlotItem(self, chart=None):
        """Get the PlotItem associated to the chart of given name.

        Parameters
        ----------
        chart : PlotDataItem, optional
            The chart whose PlotItem we will select.
            If None the default one will be selected.

        Returns
        -------
        PlotItem
            The PlotItem associated to the selected chart.
        """
        if chart is None:
            return self.pi
        else:
            return self.plot_items[chart]

    def setAxisRange(self, axis, axisRange=None, **kwargs):
        """Sets the axisRange of the axis with given name.

        Parameters
        ----------
        axis : AxisItem
            The axis to change.
        axisRange : list of int or float, optional
            The axisRange to set to the axis to selected.
            If None: axisAutoRange will be enabled.
            If list of len 1: axisRange will be set between 0 and axisRange[0]
            If list of len 2: axisRange will be set between the two axisRange values
        kwargs : dict, optional
            arguments to be passed to the ViewBox's setXRange or setYRange functions.
        """
        if axisRange is None or len(axisRange) == 0:
            # AUTORANGE
            axisRange = None
        elif len(axisRange) == 1:
            # ZERO TO R
            axisRange = [min(0, *axisRange), max(0, *axisRange)]
        elif len(axisRange) == 2:
            # SET GIVEN RANGE
            axisRange = [min(*axisRange), max(*axisRange)]
        else:
            raise AttributeError("bad axisRange")
        if axisRange is None:
            self.enableAxisAutoRange(axis)
        else:
            self.disableAxisAutoRange(axis)
            charts = axis.charts
            if axis.orientation in {"top", "bottom"}:  # IS X AXIS
                for chart in charts:
                    self.plot_items[chart].vb.setXRange(*axisRange, **kwargs)
            elif axis.orientation in {"left", "right"}:  # IS Y AXIS
                for chart in charts:
                    self.plot_items[chart].vb.setYRange(*axisRange, **kwargs)

    def update(self):
        """Updates all charts' contents."""
        for axis in self.axes:
            if axis.autorange:
                charts = axis.charts
                bounds = []
                if axis.orientation in {"top", "bottom"}:  # IS X AXIS
                    for chart in charts:
                        bounds += chart.dataBounds(ViewBox.XAxis)
                    bounds = [bound for bound in bounds if bound is not None]
                    if len(bounds) > 0:
                        for chart in charts:
                            vb = self.plot_items[chart].vb
                            vb.setXRange(min(bounds), max(bounds))
                elif axis.orientation in {"left", "right"}:  # IS Y AXIS
                    for chart in charts:
                        bounds += chart.dataBounds(ViewBox.YAxis)
                    bounds = [bound for bound in bounds if bound is not None]
                    if len(bounds) > 0:
                        for chart in charts:
                            vb = self.plot_items[chart].vb
                            vb.setYRange(min(bounds), max(bounds))
        super().update()

    # todo does this autorange stuff do anything? ah ha! it's used in PlotItem.
    def enableAxisAutoRange(self, axis=None):
        """Enables autorange for the axis with given name.

        Parameters
        ----------
        axis : AxisItem, option
            The axis for which to enable autorange, or all axes if None.
        """
        if axis is not None:
            axis.autorange = True
        else:
            for axis in self.axes:
                axis.autorange = True

    def disableAxisAutoRange(self, axis=None):
        """Disables autorange for the axis with given name.

        Parameters
        ----------
        axis : AxisItem, option
            The axis for which to disable autorange, or all axes if None.
        """
        if axis is not None:
            axis.autorange = False
        else:
            for axis in self.axes:
                axis.autorange = False
