__all__ = ["MultiAxisPlotWidget"]

import weakref

from ..graphicsItems.AxisItem import AxisItem
from ..graphicsItems.PlotDataItem import PlotDataItem
from ..graphicsItems.PlotItem.PlotItem import PlotItem
from ..graphicsItems.ViewBox import ViewBox
from ..widgets.PlotWidget import PlotWidget


class MultiAxisPlotWidget(PlotWidget):
    def __init__(self, **kargs):
        """PlotWidget but with support for multi axis."""
        PlotWidget.__init__(self, enableMenu=False, **kargs)
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

    def addAxis(self, name, *args, axis=None, **kwargs):
        """Add a new axis (AxisItem) to the widget, will be shown and linked to a chart when used in addChart().

        Parameters:
        name (str):
            The name associated with this axis item, used to store and recall the newly created axis.
            Also sets the AxisItem name parameter.
        chart AxisItem:
            The axis to be used.
            If left as None a new AxisItem will be created and it's name parameter set.
        *args and **kwargs:
            Parameters to be passed to the newly created AxisItem.
            Remember to set the AxisItem orientation.
        Returns:
            AxisItem: The newly created AxisItem.
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
        """Add a new chart (PlotDataItem in a PlotItem) to the widget, will be shown when used in makeLayout().

        Parameters:
        name (str):
            The name associated with this chart item, used to store and recall the newly created chart.
            Also sets the PlotItem name parameter and the PlotDataItem name parameter if no other chart is passed.
        x_axis (str, None):
            one of the default PlotItem axis names ("top", "right", "bottom", "left"),
            or the name of an axis previously created by calling addAxis().
            This axis will be set in the new PlotItem based on the selected axis orientation.
            If None "bottom" will be used as default.
        y_axis (str, None):
            one of the default PlotItem axis names ("top", "right", "bottom", "left"),
            or the name of an axis previously created by calling addAxis().
            This axis will be set in the new PlotItem based on the selected axis orientation.
            If None "left" will be used as default.
        chart (PlotDataItem, ...):
            The chart to be used inside the new PlotItem.
            If left as None a new PlotDataItem will be created and it's name parameter set.
        *args and **kwargs:
            Parameters to be passed to the newly created PlotItem.
        Returns:
            PlotDataItem: The newly created PlotDataItem.
            PlotItem: The newly created PlotItem.
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
            plotitem = PlotItem(parent=self.pi, name=name, *args, enableMenu=False, **kwargs)  # pass axisitems?
            # hide all plotitem axis (they vould interfere with viewbox)
            for a in ["left", "bottom", "right", "top"]:
                plotitem.hideAxis(a)
            # fix parent legend not showing child charts
            plotitem.legend = self.pi.legend
            for k, pos, axis in [["left", [2, 0], y], ["bottom", [3, 1], x],  ["right", [2, 2], y], ["top", [1, 1], x]]:
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
            del item

    def makeLayout(self, axes=None, charts=None):
        """Adds all given axes and charts to the widget.

        Parameters:
        axes (list, None):
            The names associated with the axes to show.
        charts (list, None):
            The names associated with the charts to show.
        """
        self.clearLayout()
        shown_axes = self.show_axes(axes)
        shown_charts = self.show_charts(charts)
        if len(shown_charts) != 0:
            top_level = shown_charts[list(shown_charts)[-1]]
            # FROM "https://github.com/pyqtgraph/pyqtgraph/pull/2010" by herodotus77
            # propagate mouse actions to charts "hidden" behind
            for k, c in self.charts.items():
                self.connect_signals(top_level, c)
        # MOVE LEGEND TO LAYOUT
        if self.pi.legend is not None:
            self.pi.legend.setParentItem(self.pi)
        self.update()

    def show_axes(self, axes=None):
        # SELECT AND ASSEMBLE AXIS
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
        for y, a in enumerate([None] + lo["top"] + [None] + lo["bottom"]):
            if a is not None:
                a.show()
                self.layout.addItem(a, y, vx)
                shown_axes["y"][a.name] = a
        return shown_axes

    def show_charts(self, charts=None):
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

    def connect_signals(self, top_level, chart):
        tvb = top_level.plotItem.vb
        cvb = chart.plotItem.vb
        scene = self.scene()
        for axis_name in chart.axes:
            # link axis to view
            axis = self.axes[axis_name]
            # connect view changes to axis
            # set axis main view link if not assigned
            if axis.linkedView() is None:
                axis._linkedView = weakref.ref(cvb)
            # FROM AxisItem.linkToView
            # connect view changes to axis changes
            if axis.orientation in ["right", "left"]:
                cvb.sigYRangeChanged.connect(axis.linkedViewChanged)
            elif axis.orientation in ["top", "bottom"]:
                cvb.sigXRangeChanged.connect(axis.linkedViewChanged)
            cvb.sigResized.connect(axis.linkedViewChanged)
            axis_view = axis.linkedView()
            if axis_view is not cvb and tvb is not cvb:
                # FROM ViewBox.linkView
                # connext axis's view changes to view since axis acts just like a proxy to it
                if axis.orientation in ["top", "bottom"]:
                    # connect axis main view changes to view
                    cvb.state["linkedViews"][cvb.XAxis] = weakref.ref(axis_view)
                    axis_view.sigXRangeChanged.connect(cvb.linkedXChanged)
                    axis_view.sigResized.connect(cvb.linkedXChanged)
                    # disable autorange on manual movements
                    axis_view.sigXRangeChangedManually.connect(lambda mask: self.disableAxisAutoRange(axis_name))
                elif axis.orientation in ["right", "left"]:
                    # connect axis main view changes to view
                    cvb.state["linkedViews"][cvb.YAxis] = weakref.ref(axis_view)
                    axis_view.sigYRangeChanged.connect(cvb.linkedYChanged)
                    axis_view.sigResized.connect(cvb.linkedYChanged)
                    # disable autorange on manual movements
                    axis_view.sigYRangeChangedManually.connect(lambda mask: self.disableAxisAutoRange(axis_name))
            cvb.sigStateChanged.emit(cvb)
        # resize plotitem according to the master one
        # resizing it's view doesn't work for some reason
        self.vb.sigResized.connect(lambda vb: chart.plotItem.setGeometry(vb.sceneBoundingRect()))
        # fix prepareForPaint by outofculture
        scene.sigPrepareForPaint.connect(cvb.prepareForPaint)
        try:
            cvb.sigMouseDragged.disconnect()
        except (TypeError, RuntimeError):
            pass
        try:
            cvb.sigMouseWheel.disconnect()
        except (TypeError, RuntimeError):
            pass
        try:
            cvb.sigHistoryChanged.disconnect()
        except (TypeError, RuntimeError):
            pass
        if cvb is not tvb:
            # FROM "https://github.com/pyqtgraph/pyqtgraph/pull/2010" by herodotus77
            # propagate mouse actions to charts "hidden" behind
            tvb.sigMouseDragged.connect(cvb.mouseDragEvent)
            tvb.sigMouseWheel.connect(cvb.wheelEvent)
            tvb.sigHistoryChanged.connect(cvb.scaleHistory)

    def clean(self):
        """Clears all charts' contents."""
        # CLEAR PLOTS
        for p in self.charts.values():
            p.clear()
        self.update()

    def getPlotItem(self, name=None):
        """Get the PlotItem associated to the chart of given name.

        Parameters:
        name (str, None):
            The name of the chart to select.
            If None the default one will be selected.
        Returns:
            PlotItem: The PlotItem associated to the selected chart.
        """
        if name is None:
            return self.pi
        else:
            return self.charts[name].plotItem

    def setAxisRange(self, axis_name, range=None, **kwargs):
        """Sets the range of the axis with given name.

        Parameters:
        axis_name (str, None):
            The name of the axis to select.
        range (list, None):
            The range to set to the axis to selected.
            If None: autorange will be enabled.
            If list of len 1: range will be set between 0 and range[0]
            If list of len 2: range will be set between the two range values
        """
        if range is None or len(range) == 0:
            # AUTORANGE
            range = None
        elif len(range) == 1:
            # ZERO TO R
            range = [min(0, *range), max(0, *range)]
        elif len(range) == 2:
            # SET GIVEN RANGE
            range = [min(*range), max(*range)]
        else:
            raise AttributeError("bad range")
        if range is None:
            self.enableAxisAutoRange(axis_name)
        else:
            self.disableAxisAutoRange(axis_name)
            axis = self.axes[axis_name]
            charts = [self.charts[chart] for chart in axis.charts]
            if axis.orientation in ["top", "bottom"]:  # IS X AXIS
                for chart in charts:
                    vb = chart.plotItem.vb
                    vb.setXRange(*range, **kwargs)
            elif axis.orientation in ["left", "right"]:  # IS Y AXIS
                for chart in charts:
                    vb = chart.plotItem.vb
                    vb.setYRange(*range, **kwargs)

    def update(self):
        """Updates all charts' contents."""
        for axis_name, axis in self.axes.items():
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

    def enableAxisAutoRange(self, axis_name):
        """Enables autorange for the axis with given name.

        Parameters:
        axis_name (str, None):
            The name of the axis to select.
        """
        self.axes[axis_name].autorange = True

    def disableAxisAutoRange(self, axis_name):
        """Disables autorange for the axis with given name.

        Parameters:
        axis_name (str, None):
            The name of the axis to select.
        """
        self.axes[axis_name].autorange = False
