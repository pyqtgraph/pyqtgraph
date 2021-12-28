__all__ = ["MultiAxisPlotWidget"]

import weakref

from ..functions import connect_lambda
from ..graphicsItems.AxisItem import AxisItem
from ..graphicsItems.PlotDataItem import PlotDataItem
from ..graphicsItems.PlotItem.PlotItem import PlotItem
from ..graphicsItems.ViewBox import ViewBox
from ..Qt.QtCore import QObject
from ..widgets.PlotWidget import PlotWidget


class MultiAxisPlotWidget(PlotWidget):
    def __init__(self, **kargs):
        """PlotWidget but with support for multi axis.
        usage: you generally want to call in this order:
        .addAxis(...) as many times as axes needed,
        .addChart(...) as many times as charts needed,
        and then .makeLayout() once all the needed elements have been added.
        Also consider calling .update() after updating the chart data.
        Refer to the example named "MultiAxisPlotWidget example" under the "Widgets" section if needed"""
        super().__init__(enableMenu=False, **kargs)
        # plotitem shortcut
        self.pi = super().getPlotItem()
        # override autorange button behaviour
        self.pi.autoBtn.clicked.disconnect()
        connect_lambda(self.pi.autoBtn.clicked, self, lambda self, button: (self.enableAxisAutoRange(), self.update()))
        # default vb from plotItem shortcut
        self.vb = self.pi.vb
        # layout shortcut
        self.layout = self.pi.layout
        # CHARTS
        self.axes = {}
        # select custom behaviour
        self.pi.extra_axes = self.axes
        self.charts = {}
        self.plot_items = {}
        self._signalConnectionsByChart = {}

    def addAxis(self, name, *args, axis=None, **kwargs):
        """Add a new axis (AxisItem) to the widget, will be shown and linked to a
        chart when used in addChart().

        Parameters
        ----------
        name : str
            The name associated with this axis item, used to store and recall
            the newly created axis. Also sets the AxisItem name parameter.
        axis : AxisItem, optional
            The axis to be used. If left as None a new AxisItem will be created
            and it's name parameter set.
        args : iterable, optional
            arguments to be passed to the newly created AxisItem
        kwargs : dict, optional
            arguments to be passed to the newly created AxisItem.  Remember to
            set the AxisItem orientation.

        Returns
        -------
        AxisItem
            The newly created AxisItem.
        """
        if name is None:
            raise AssertionError("Axis name should not be None")
        if axis is None:
            axis = AxisItem(*args, **kwargs)
        axis.name = name
        axis.autorange = True
        axis.charts = []
        self.axes[name] = axis
        axis.showLabel(True)
        return axis

    def addChart(self, name, xAxisName="bottom", yAxisName="left", chart=None, *args, **kwargs):
        """Add a new chart (PlotDataItem in a PlotItem) to the widget, will be
        shown when used in makeLayout().

        Parameters
        ----------
        name : str
            The name associated with this chart item, used to store and recall
            the newly created chart. Also sets the PlotItem name parameter and
            the PlotDataItem name parameter if no other chart is passed.
        xAxisName : str, optional
            one of the default PlotItem axis names ("top", "right", "bottom",
            "left"), or the name of an axis previously created by calling
            addAxis(). This axis will be set in the new PlotItem based on the
            selected axis orientation. If omitted "bottom" will be used as
            default.
        yAxisName : str, optional
            one of the default PlotItem axis names ("top", "right", "bottom",
            "left"), or the name of an axis previously created by calling
            addAxis(). This axis will be set in the new PlotItem based on the
            selected axis orientation. If omitted "left" will be used as default.
        chart : PlotDataItem, optional
            The chart to be used inside the new PlotItem. If left as None a
            new PlotDataItem will be created and it's name parameter set.
        args : iterable, optional
            Arugments to be passed to the newly created PlotItem. See
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
        if name is None:
            raise AssertionError("Chart name should not be None")
        if xAxisName not in self.axes:
            if xAxisName in {"bottom", "top"}:
                # add default axis to the list of axes if requested
                x_axis = self.addAxis(xAxisName, axis=self.pi.axes[xAxisName]["item"])
            else:
                x_axis = self.addAxis(xAxisName, "bottom")
        else:
            x_axis = self.axes[xAxisName]
        if yAxisName not in self.axes:
            if yAxisName in {"left", "right"}:
                # add default axis to the list of axes if requested
                y_axis = self.addAxis(yAxisName, yAxisName)
            else:
                y_axis = self.addAxis(yAxisName, "left")
        else:
            y_axis = self.axes[yAxisName]
        if xAxisName in {"bottom", "top"} and yAxisName in {"left", "right"}:
            # use default plotitem if none provided
            plotitem = self.pi
        else:
            # VIEW
            plotitem = PlotItem(parent=self.pi, name=name, *args,
                                enableMenu=False, **kwargs)  # pass axisitems?
            # disable buttons (autoranging)
            plotitem.hideButtons()
            # fix parent legend not showing child charts
            plotitem.legend = self.pi.legend
            for axis_orientation, pos, axis in [["left", [2, 0], y_axis], ["bottom", [3, 1], x_axis], ["right", [2, 2], y_axis], ["top", [1, 1], x_axis]]:
                # # DO NOT USE, WILL MAKE AXIS UNMATCHED TO DATA
                # # you can't add the new ones after, it doesn't work for some reason
                # # hide them instead
                # plotitem.layout.removeItem(plotitem.layout.itemAt(*pos))
                # hide all plotitem axis (they vould interfere with viewbox)
                plotitem.hideAxis(axis_orientation)
                if axis.orientation == axis_orientation:
                    plotitem.axes[axis_orientation] = {"item": axis, "pos": pos}
        # CHART
        if chart is None:
            chart = PlotDataItem(name=name)
        plotitem.addItem(chart)
        # keep plotitem inside chart
        self.plot_items[chart.name] = plotitem
        # keeptrack of connections
        if chart.name not in self._signalConnectionsByChart:
            self._signalConnectionsByChart[chart.name] = {}
        # keep axis track
        chart.axes = [xAxisName, yAxisName]
        # keep chart
        self.charts[name] = chart
        # create a mapping for this chart and his axis
        self.axes[xAxisName].charts.append(name)
        self.axes[yAxisName].charts.append(name)
        return chart, plotitem

    def clearLayout(self):
        """Clears the widget but keeps all created axis and charts."""
        while self.layout.count() > 0:
            item = self.layout.itemAt(0)
            self.layout.removeAt(0)
            self.scene().removeItem(item)
        for chart in self.charts.values():
            self._disconnect_all(chart)

    def makeLayout(self, axes=None, charts=None):
        """Adds all given axes and charts to the widget.

        Parameters
        ----------
        axes : list of str, optional
            The names associated with the axes to show.
            Axes will be ordered from left-to-right and from top-to-bottom
            following the given axes list order if given,
            or the axis creation order otherwise.
        charts : list of PlotItems, optional
            The names associated with the charts to show.
        """
        self.clearLayout()
        _ = self._show_axes(axes)
        shown_charts = self._show_charts(charts)
        if len(shown_charts) != 0:
            for chart in self.charts.values():
                self._connect_signals(chart)
        self.update()

    def _show_axes(self, axes=None):
        """Shows all the selected axes."""
        # SELECT AND ASSEMBLE AXES
        if axes is None:
            axes = self.axes.keys()
        axes = list(dict.fromkeys(axes))
        lo = {
            "left": [],
            "right": [],
            "top": [],
            "bottom": [],
        }
        for axis in [self.axes[axis_name] for axis_name in axes]:
            # add margin to the axis label to avoid overlap of other axis ticks
            axis.setStyle(labelMargin=[5, 8])
            lo[axis.orientation].append(axis)
        horizontal_axes_pos_x = len(lo["left"])
        vertical_axes_pos_y = 1 + len(lo["top"])
        # ADD TITLE ON TOP
        self.pi.titleLabel.show()
        self.layout.addItem(self.pi.titleLabel, 0, horizontal_axes_pos_x)
        # ADD MAIN PLOTITEM
        self.vb.show()
        self.layout.addItem(self.vb, vertical_axes_pos_y, horizontal_axes_pos_x)
        # ADD AXIS
        shown_axes = {"x": {}, "y": {}}
        for axes_pos_x, axis in enumerate(lo["left"] + [None] + lo["right"]):
            if axis is not None:
                axis.show()
                self.layout.addItem(axis, vertical_axes_pos_y, axes_pos_x)
                shown_axes["x"][axis.name] = axis
        for axes_pos_y, axis in enumerate(lo["top"] + [None] + lo["bottom"], start=1):
            if axis is not None:
                axis.show()
                self.layout.addItem(axis, axes_pos_y, horizontal_axes_pos_x)
                shown_axes["y"][axis.name] = axis
        return shown_axes

    def _show_charts(self, charts=None):
        """Shows all the selected charts."""
        # SELECT CHARTS
        if charts is None:
            charts = self.charts.keys()
        charts = list(dict.fromkeys(charts))
        for chart in [self.charts[chart_name] for chart_name in self.charts.keys() - set(charts)]:
            chart.hide()
        shown_charts = {chart_name: self.charts[chart_name] for chart_name in charts}
        for chart_name, chart in shown_charts.items():
            chart.show()
        return shown_charts

    def _connect_signals(self, chart):
        """Connects all signals related to this widget for the given chart given the top level one."""
        self._disconnect_all(chart)
        chart_vb = self.plot_items[chart.name].vb
        signals = self._signalConnectionsByChart[chart.name]
        scene = self.scene()
        for axis_name in chart.axes:
            # link axis to view
            axis = self.axes[axis_name]
            # connect view changes to axis
            # set axis main view link if not assigned
            if axis.linkedView() is None:
                axis._linkedView = weakref.ref(chart_vb)
            # FROM AxisItem.linkToView
            # connect view changes to axis changes
            if axis.orientation in {"right", "left"}:
                signals["propagate chart range to axis"] = chart_vb.sigYRangeChanged.connect(
                    axis.linkedViewChanged)
            elif axis.orientation in {"top", "bottom"}:
                signals["propagate chart range to axis"] = chart_vb.sigXRangeChanged.connect(
                    axis.linkedViewChanged)
            signals["propagate chart resize to axis"] = chart_vb.sigResized.connect(
                axis.linkedViewChanged)
            axis_view = axis.linkedView()
            if chart_vb is not axis_view:
                # FROM ViewBox.linkView
                # connext axis's view changes to view since axis acts just like a proxy to it
                if axis.orientation in {"top", "bottom"}:
                    # connect axis main view changes to view
                    chart_vb.state["linkedViews"][chart_vb.XAxis] = weakref.ref(axis_view)
                    # this signal is received multiple times when using mouse actions directly on the viewbox
                    # this causes the non top layer views to scroll more than the frontmost one
                    signals["propagate axis range to chart"] = axis_view.sigXRangeChanged.connect(
                        chart_vb.linkedXChanged)
                    signals["propagate axis resize to chart"] = axis_view.sigResized.connect(
                        chart_vb.linkedXChanged)
                    # disable autorange on manual movements
                    # using connect_lambda here as a workaround
                    # refere to the documentation of connect_lambda in functions.py for an explaination of the issue
                    # signals["disable axis autorange on axis manual change"] =
                    #     axis_view.sigXRangeChangedManually.connect(lambda mask: self.disableAxisAutoRange(axis_name))  # THIS BREAKS GARBAGE COLLECTION
                    signals["disable axis autorange on axis manual change"] = connect_lambda(axis_view.sigXRangeChangedManually, self, lambda self, mask: self.disableAxisAutoRange(axis_name))
                elif axis.orientation in {"right", "left"}:
                    # connect axis main view changes to view
                    chart_vb.state["linkedViews"][chart_vb.YAxis] = weakref.ref(axis_view)
                    # this signal is received multiple times when using mouse actions directly on the viewbox
                    # this causes the non top layer views to scroll more than the frontmost one
                    signals["propagate axis range to chart"] = axis_view.sigYRangeChanged.connect(chart_vb.linkedYChanged)
                    signals["propagate axis resize to chart"] = axis_view.sigResized.connect(
                        chart_vb.linkedYChanged)
                    # disable autorange on manual movements
                    # using connect_lambda here as a workaround
                    # refere to the documentation of connect_lambda in functions.py for an explaination of the issue
                    # signals["disable axis autorange on axis manual change"] =
                    #     axis_view.sigYRangeChangedManually.connect(lambda mask: self.disableAxisAutoRange(axis_name))  # THIS BREAKS GARBAGE COLLECTION
                    signals["disable axis autorange on axis manual change"] = connect_lambda(axis_view.sigYRangeChangedManually, self, lambda self, mask: self.disableAxisAutoRange(axis_name))
            chart_vb.sigStateChanged.emit(chart_vb)
        # resize plotitem according to the master one
        # resizing it's view doesn't work for some reason
        if chart_vb is not self.vb:
            # make default vb always the top level one chart vb
            chart_vb.setZValue(0)
            self.vb.setZValue(9999)  # over 9thousand!
            chart_pi = self.plot_items[chart.name]
            # using connect_lambda here as a workaround
            # refere to the documentation of connect_lambda in functions.py for an explaination of the issue
            signals["propagate default vb resize to chart"] = connect_lambda(self.vb.sigResized, chart_pi, lambda chart_pi, vb: chart_pi.setGeometry(vb.sceneBoundingRect()))
        # fix prepareForPaint by outofculture
        signals["propagate default scene prepare_for_paint to chart"] = scene.sigPrepareForPaint.connect(
            chart_vb.prepareForPaint)
        if chart_vb is not self.vb:
            # FROM "https://github.com/pyqtgraph/pyqtgraph/pull/2010" by herodotus77
            # propagate mouse actions to charts "hidden" behind
            signals["propagate top level mouse drag interaction to chart"] = self.vb.sigMouseDragged.connect(
                chart_vb.mouseDragEvent)
            signals["propagate top level mouse wheel interaction to chart"] = self.vb.sigMouseWheel.connect(
                chart_vb.wheelEvent)
            signals["propagate top level history changes interaction to chart"] = self.vb.sigHistoryChanged.connect(
                chart_vb.scaleHistory)

    def _disconnect_all(self, chart):
        """Disconnects all signals related to this widget for the given chart."""
        signals = self._signalConnectionsByChart[chart.name]
        for conn_name, conn in signals.items():
            if conn is not None:
                QObject.disconnect(conn)
                signals[conn_name] = None

    def clean(self):
        """Clears all charts' contents."""
        # CLEAR PLOTS
        for chart in self.charts.values():
            chart.clear()
        self.update()

    def getPlotItem(self, name=None):
        """Get the PlotItem associated to the chart of given name.

        Parameters
        ----------
        name : str, optional
            The name of the chart to select.
            If None the default one will be selected.

        Returns
        -------
        PlotItem
            The PlotItem associated to the selected chart.
        """
        if name is None:
            return self.pi
        else:
            return self.plot_items[self.charts[name].name]

    def setAxisRange(self, axisName, axisRange=None, **kwargs):
        """Sets the axisRange of the axis with given name.

        Parameters
        ----------
        axisName : str
            The name of the axis to select.
        axisRange : list of int or float, optional
            The axisRange to set to the axis to selected.
            If None: axisRutorange will be enabled.
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
            self.enableAxisAutoRange(axisName)
        else:
            self.disableAxisAutoRange(axisName)
            axis = self.axes[axisName]
            charts = [self.charts[chart] for chart in axis.charts]
            if axis.orientation in {"top", "bottom"}:  # IS X AXIS
                for chart in charts:
                    self.plot_items[chart.name].vb.setXRange(*axisRange, **kwargs)
            elif axis.orientation in {"left", "right"}:  # IS Y AXIS
                for chart in charts:
                    self.plot_items[chart.name].vb.setYRange(*axisRange, **kwargs)

    def update(self):
        """Updates all charts' contents."""
        for axis in self.axes.values():
            if axis.autorange:
                charts = [self.charts[chart] for chart in axis.charts]
                bounds = []
                if axis.orientation in {"top", "bottom"}:  # IS X AXIS
                    for chart in charts:
                        bounds += chart.dataBounds(ViewBox.XAxis)
                    bounds = [bound for bound in bounds if bound is not None]
                    if len(bounds) > 0:
                        for chart in charts:
                            vb = self.plot_items[chart.name].vb
                            vb.setXRange(min(bounds), max(bounds))
                elif axis.orientation in {"left", "right"}:  # IS Y AXIS
                    for chart in charts:
                        bounds += chart.dataBounds(ViewBox.YAxis)
                    bounds = [bound for bound in bounds if bound is not None]
                    if len(bounds) > 0:
                        for chart in charts:
                            vb = self.plot_items[chart.name].vb
                            vb.setYRange(min(bounds), max(bounds))
        super().update()

    def enableAxisAutoRange(self, axisName=None):
        """Enables autorange for the axis with given name.

        Parameters
        ----------
        axisName : str
            The name of the axis to select.
        """
        if axisName is not None:
            self.axes[axisName].autorange = True
        else:
            for axis in self.axes.values():
                axis.autorange = True

    def disableAxisAutoRange(self, axisName=None):
        """Disables autorange for the axis with given name.

        Parameters
        ----------
        axisName : str
            The name of the axis to select.
        """
        if axisName is not None:
            self.axes[axisName].autorange = False
        else:
            for axis in self.axes.values():
                axis.autorange = False
