__all__ = ["MultiAxisPlotWidget"]

import weakref

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
        # default vb from plotItem shortcut
        self.vb = self.pi.vb
        # layout shortcut
        self.layout = self.pi.layout
        # hide default axis
        for a in ["left", "bottom", "right", "top"]:
            self.pi.hideAxis(a)
        # CHARTS
        self.axes = {}
        self.charts = {}
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
        if axis is None:
            axis = AxisItem(*args, **kwargs)
        axis.name = name
        axis.autorange = True
        axis.charts = []
        self.axes[name] = axis
        axis.showLabel(True)
        return axis

    def addChart(self, name, x_axis="bottom", y_axis="left", chart=None, *args, **kwargs):
        """Add a new chart (PlotDataItem in a PlotItem) to the widget, will be
        shown when used in makeLayout().

        Parameters
        ----------
        name : str
            The name associated with this chart item, used to store and recall
            the newly created chart. Also sets the PlotItem name parameter and
            the PlotDataItem name parameter if no other chart is passed.
        x_axis : str, optional
            one of the default PlotItem axis names ("top", "right", "bottom",
            "left"), or the name of an axis previously created by calling
            addAxis(). This axis will be set in the new PlotItem based on the
            selected axis orientation. If None "bottom" will be used as
            default.
        y_axis : str, optional
            one of the default PlotItem axis names ("top", "right", "bottom",
            "left"), or the name of an axis previously created by calling
            addAxis(). This axis will be set in the new PlotItem based on the
            selected axis orientation. If None "left" will be used as default.
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
        # add default axis to the list of axes if requested
        if x_axis in ["bottom", "top"]:
            x = self.addAxis(x_axis, axis=self.pi.axes[x_axis]["item"])
        elif x_axis not in self.axes:
            x = self.addAxis(x_axis)
        else:
            x = self.axes[x_axis]
        if y_axis in ["left", "right"]:
            y = self.addAxis(x_axis, axis=self.pi.axes[y_axis]["item"])
        elif y_axis not in self.axes:
            y = self.addAxis(y_axis)
        else:
            y = self.axes[y_axis]
        if x_axis in ["bottom", "top"] and y_axis in ["left", "right"]:
            # use default plotitem if none provided
            plotitem = self.pi
        else:
            # VIEW
            plotitem = PlotItem(parent=self.pi, name=name, *args,
                                enableMenu=False, **kwargs)  # pass axisitems?
            # hide all plotitem axis (they vould interfere with viewbox)
            for a in ["left", "bottom", "right", "top"]:
                plotitem.hideAxis(a)
            # fix parent legend not showing child charts
            plotitem.legend = self.pi.legend
            for k, pos, axis in [["left", [2, 0], y], ["bottom", [3, 1], x], ["right", [2, 2], y], ["top", [1, 1], x]]:
                # # DO NOT USE, WILL MAKE AXIS UNMATCHED TO DATA
                # # you can't add the new ones after, it doesn't work for some reason
                # # hide them instead
                # plotitem.layout.removeItem(plotitem.layout.itemAt(*pos))
                if axis.orientation == k:
                    plotitem.axes[k] = {"item": axis, "pos": pos}
                # hide plotitem axes
                plotitem.hideAxis(a)
        # CHART
        if chart is None:
            chart = PlotDataItem(name=name)
        plotitem.addItem(chart)
        # keep plotitem inside chart
        chart.plotItem = plotitem
        # keeptrack of connections
        if chart.name not in self._signalConnectionsByChart:
            self._signalConnectionsByChart[chart.name] = {}
        # keep axis track
        chart.axes = [x_axis, y_axis]
        # keep chart
        self.charts[name] = chart
        # create a mapping for this chart and his axis
        self.axes[x_axis].charts.append(name)
        self.axes[y_axis].charts.append(name)
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
        charts : list of PlotItems, optional
            The names associated with the charts to show.
        """
        self.clearLayout()
        _ = self._show_axes(axes)
        shown_charts = self._show_charts(charts)
        if len(shown_charts) != 0:
            top_level = shown_charts[list(shown_charts)[-1]]
            # FROM "https://github.com/pyqtgraph/pyqtgraph/pull/2010" by herodotus77
            # propagate mouse actions to charts "hidden" behind
            for k, c in self.charts.items():
                self._connect_signals(top_level, c)
        # MOVE LEGEND TO LAYOUT
        if self.pi.legend is not None:
            self.pi.legend.setParentItem(self.pi)
        self.update()

    def _show_axes(self, axes=None):
        """Shows all the selected axes."""
        # SELECT AND ASSEMBLE AXES
        if axes is None:
            axes = list(self.axes)
        lo = {
            "left": [],
            "right": [],
            "top": [],
            "bottom": [],
        }
        for k, a in self.axes.items():
            if k in axes:
                lo[a.orientation].append(a)
        vx = len(lo["left"])
        vy = 1 + len(lo["top"])
        # ADD TITLE ON TOP
        self.pi.titleLabel.show()
        self.layout.addItem(self.pi.titleLabel, 0, vx)
        # ADD MAIN PLOTITEM
        self.vb.show()
        self.layout.addItem(self.vb, vy, vx)
        # ADD AXIS
        shown_axes = {"x": {}, "y": {}}
        for x, a in enumerate(lo["left"] + [None] + lo["right"]):
            if a is not None:
                a.show()
                self.layout.addItem(a, vy, x)
                shown_axes["x"][a.name] = a
        for y, a in enumerate(lo["top"] + [None] + lo["bottom"], start=1):
            if a is not None:
                a.show()
                self.layout.addItem(a, y, vx)
                shown_axes["y"][a.name] = a
        return shown_axes

    def _show_charts(self, charts=None):
        """Shows all the selected charts."""
        # SELECT CHARTS
        if charts is None:
            charts = self.charts
        shown_charts = {}
        for k, c in self.charts.items():
            if k in charts:
                c.show()
                shown_charts[k] = c
            else:
                c.hide()
        return shown_charts

    def _connect_signals(self, top_level, chart):
        """Connects all signals related to this widget for the given chart given the top level one."""
        self._disconnect_all(chart)
        top_vb = top_level.plotItem.vb
        chart_vb = chart.plotItem.vb
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
            if axis.orientation in ["right", "left"]:
                signals["chart_vb.sigYRangeChanged"] = chart_vb.sigYRangeChanged.connect(
                    axis.linkedViewChanged)
            elif axis.orientation in ["top", "bottom"]:
                signals["chart_vb.sigXRangeChanged"] = chart_vb.sigXRangeChanged.connect(
                    axis.linkedViewChanged)
            signals["chart_vb.sigResized"] = chart_vb.sigResized.connect(
                axis.linkedViewChanged)
            axis_view = axis.linkedView()
            if chart_vb is not axis_view:
                # FROM ViewBox.linkView
                # connext axis's view changes to view since axis acts just like a proxy to it
                if axis.orientation in ["top", "bottom"]:
                    # connect axis main view changes to view
                    chart_vb.state["linkedViews"][chart_vb.XAxis] = weakref.ref(axis_view)
                    # this signal is received multiple times when using mouse actions directly on the viewbox
                    # this causes the non top layer views to scroll more than the frontmost one
                    signals["axis_view.sigXRangeChanged"] = axis_view.sigXRangeChanged.connect(
                        chart_vb.linkedXChanged)
                    signals["axis_view.sigResized"] = axis_view.sigResized.connect(
                        chart_vb.linkedXChanged)
                    # disable autorange on manual movements
                    signals["axis_view.sigXRangeChangedManually"] = axis_view.sigXRangeChangedManually.connect(
                        lambda mask: self.disableAxisAutoRange(axis_name))
                elif axis.orientation in ["right", "left"]:
                    # connect axis main view changes to view
                    chart_vb.state["linkedViews"][chart_vb.YAxis] = weakref.ref(axis_view)
                    # this signal is received multiple times when using mouse actions directly on the viewbox
                    # this causes the non top layer views to scroll more than the frontmost one
                    signals["axis_view.sigYRangeChanged"] = axis_view.sigYRangeChanged.connect(
                        chart_vb.linkedYChanged)
                    signals["axis_view.sigResized"] = axis_view.sigResized.connect(
                        chart_vb.linkedYChanged)
                    # disable autorange on manual movements
                    signals["axis_view.sigYRangeChangedManually"] = axis_view.sigYRangeChangedManually.connect(
                        lambda mask: self.disableAxisAutoRange(axis_name))
            chart_vb.sigStateChanged.emit(chart_vb)
        # resize plotitem according to the master one
        # resizing it's view doesn't work for some reason
        signals["self.vb.sigResized"] = self.vb.sigResized.connect(
            lambda vb: chart.plotItem.setGeometry(vb.sceneBoundingRect()))
        # fix prepareForPaint by outofculture
        signals["scene.sigPrepareForPaint"] = scene.sigPrepareForPaint.connect(
            chart_vb.prepareForPaint)
        if chart_vb is not top_vb:
            # FROM "https://github.com/pyqtgraph/pyqtgraph/pull/2010" by herodotus77
            # propagate mouse actions to charts "hidden" behind
            signals["top_vb.sigMouseDragged"] = top_vb.sigMouseDragged.connect(
                chart_vb.mouseDragEvent)
            signals["top_vb.sigMouseWheel"] = top_vb.sigMouseWheel.connect(
                chart_vb.wheelEvent)
            signals["top_vb.sigHistoryChanged"] = top_vb.sigHistoryChanged.connect(
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
        for p in self.charts.values():
            p.clear()
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
            return self.charts[name].plotItem

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
            if axis.orientation in ["top", "bottom"]:  # IS X AXIS
                for chart in charts:
                    vb = chart.plotItem.vb
                    vb.setXRange(*axisRange, **kwargs)
            elif axis.orientation in ["left", "right"]:  # IS Y AXIS
                for chart in charts:
                    vb = chart.plotItem.vb
                    vb.setYRange(*axisRange, **kwargs)

    def update(self):
        """Updates all charts' contents."""
        for axis in self.axes.values():
            if axis.autorange:
                charts = [self.charts[chart] for chart in axis.charts]
                bounds = []
                if axis.orientation in ["top", "bottom"]:  # IS X AXIS
                    for chart in charts:
                        bounds += chart.dataBounds(ViewBox.XAxis)
                    bounds = [bound for bound in bounds if bound is not None]
                    if len(bounds) > 0:
                        for chart in charts:
                            vb = chart.plotItem.vb
                            vb.setXRange(min(bounds), max(bounds))
                elif axis.orientation in ["left", "right"]:  # IS Y AXIS
                    for chart in charts:
                        bounds += chart.dataBounds(ViewBox.YAxis)
                    bounds = [bound for bound in bounds if bound is not None]
                    if len(bounds) > 0:
                        for chart in charts:
                            vb = chart.plotItem.vb
                            vb.setYRange(min(bounds), max(bounds))
        super().update()

    def enableAxisAutoRange(self, axisName):
        """Enables autorange for the axis with given name.

        Parameters
        ----------
        axisName : str
            The name of the axis to select.
        """
        self.axes[axisName].autorange = True

    def disableAxisAutoRange(self, axisName):
        """Disables autorange for the axis with given name.

        Parameters
        ----------
        axisName : str
            The name of the axis to select.
        """
        self.axes[axisName].autorange = False
