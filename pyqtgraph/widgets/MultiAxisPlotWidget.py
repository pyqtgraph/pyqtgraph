# -*- coding: utf-8 -*-
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
        self.vb = self.pi.getViewBox()
        # layout shortcut
        self.layout = self.pi.layout
        # hide default axis
        for a in ["left", "bottom", "right", "top"]:
            self.pi.hideAxis(a)
        # CHARTS
        self.axis = {}
        self.charts = {}

    def addAxis(self, name, *args, **kwargs):
        """Add a new axis (AxisItem) to the widget, will be shown and linked to a chart when used in addChart().

        Parameters:
        name (str):
            The name associated with this axis item, used to store and recall the newly created axis.
            Also sets the AxisItem name parameter.
        *args and **kwargs:
            Parameters to be passed to the newly created AxisItem.
            Remember to set the AxisItem orientation.
        Returns:
            AxisItem: The newly created AxisItem.
        """
        axis = AxisItem(name=name, *args, **kwargs)
        axis.autorange = True
        axis.charts_connections = []
        self.axis[name] = axis
        return axis

    def addChart(self, name, x_axis=None, y_axis=None, chart=None, *args, **kwargs):
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
        # CHART
        if chart is None:
            chart = PlotDataItem(name=name)
        if x_axis is None and y_axis is None:
            # use default plotitem if none provided
            plotitem = self.pi
        else:
            # Create and place axis items if necessary
            # X AXIS
            if x_axis is None:  # use default axis if none provided
                x_axis = "bottom"
            if x_axis in self.axis:
                x = self.axis[x_axis]
            else:
                self.addAxis(x_axis, "bottom", parent=self.pi)
                x = self.axis[x_axis]
            # Y AXIS
            if y_axis is None:  # use default axis if none provided
                y_axis = "left"
            if y_axis in self.axis:
                y = self.axis[y_axis]
            else:
                self.addAxis(y_axis, "left", parent=self.pi)
                y = self.axis[y_axis]
            # VIEW
            plotitem = PlotItem(parent=self.pi, name=name, *args, enableMenu=False, **kwargs)
            # hide all plotitem axis (they vould interfere with viewbox)
            for a in ["left", "bottom", "right", "top"]:
                plotitem.hideAxis(a)
            # link axis to view
            view = plotitem.getViewBox()
            self.linkAxisToView(x_axis, view)
            self.linkAxisToView(y_axis, view)
            # TODO: check this
            for k, pos, axis in [["top", [1, 1], y], ["bottom", [3, 1], x]]:  # , ["left", [2, 0], y], ["right", [2, 2], x]]:
                # # DO NOT USE, WILL MAKE AXIS UNMATCHED TO DATA
                # # you can't add the new ones after, it doesn't work for some reason
                # # hide them instead
                # plotitem.layout.removeItem(plotitem.layout.itemAt(*pos))
                plotitem.axes[k] = {"item": axis, "pos": pos}
            # fix parent legend not showing child charts
            plotitem.legend = self.pi.legend
            # resize plotitem according to the master one
            # resizing it's view doesn't work for some reason
            self.vb.sigResized.connect(lambda vb: plotitem.setGeometry(vb.sceneBoundingRect()))
        plotitem.addItem(chart)
        # keep plotitem inside chart
        chart.plotItem = plotitem
        # keep axis track
        chart.axis = [x_axis, y_axis]
        # keep chart
        self.charts[name] = chart
        # create a mapping for this chart and his axis
        self.axis[x_axis].charts_connections.append(name)
        self.axis[y_axis].charts_connections.append(name)
        return chart, plotitem

    def clearLayout(self):
        """Clears the widget but keeps all created axis and charts."""
        while self.layout.count() > 0:
            item = self.layout.itemAt(0)
            self.layout.removeAt(0)
            self.scene().removeItem(item)
            del item

    def linkAxisToView(self, axis_name, view):
        """Links an axis to a view that should use such axis.
            This is an internal function and is not intended to be used by the user.

        Parameters:
        axis_name (str):
            The name associated with the axis to link to whe view.
        view (ViewBox):
            The ViewBox to associate the axis to.
        """
        axis = self.axis[axis_name]
        # connect view changes to axis
        # set axis main view link if not assigned
        if axis.linkedView() is None:
            axis._linkedView = weakref.ref(view)
        # FROM AxisItem.linkToView
        # connect view changes to axis changes
        if axis.orientation in ["right", "left"]:
            view.sigYRangeChanged.connect(axis.linkedViewChanged)
        elif axis.orientation in ["top", "bottom"]:
            view.sigXRangeChanged.connect(axis.linkedViewChanged)
        view.sigResized.connect(axis.linkedViewChanged)
        axis_view = axis.linkedView()
        if axis_view is not view:
            # FROM ViewBox.linkView
            # connext axis's view changes to view since axis acts just like a proxy to it
            if axis.orientation in ["top", "bottom"]:
                # connect axis main view changes to view
                view.state["linkedViews"][view.XAxis] = weakref.ref(axis_view)
                axis_view.sigXRangeChanged.connect(view.linkedXChanged)
                axis_view.sigResized.connect(view.linkedXChanged)
                # disable autorange on manual movements
                axis_view.sigXRangeChangedManually.connect(lambda mask: self.disableAxisAutoRange(axis_name))
            elif axis.orientation in ["right", "left"]:
                # connect axis main view changes to view
                view.state["linkedViews"][view.YAxis] = weakref.ref(axis_view)
                axis_view.sigYRangeChanged.connect(view.linkedYChanged)
                axis_view.sigResized.connect(view.linkedYChanged)
                # disable autorange on manual movements
                axis_view.sigYRangeChangedManually.connect(lambda mask: self.disableAxisAutoRange(axis_name))
        view.sigStateChanged.emit(view)

    def makeLayout(self, axis=None, charts=None):
        """Adds all given axis and charts to the widget.

        Parameters:
        axis (list, None):
            The name associated with the axis to link to whe view.
        charts (ViewBox):
            The ViewBox to associate the axis to.
        """
        self.clearLayout()
        # SELECT AND ASSEMBLE AXIS
        if axis is None:
            axis = list(self.axis)
        lo = {
            "left": [],
            "right": [],
            "top": [],
            "bottom": [],
        }
        for k, a in self.axis.items():
            if k in axis:
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
        for x, a in enumerate(lo["left"] + [None] + lo["right"]):
            if a is not None:
                a.show()
                self.layout.addItem(a, vy, x)
        for y, a in enumerate([None] + lo["top"] + [None] + lo["bottom"]):
            if a is not None:
                a.show()
                self.layout.addItem(a, y, vx)
        # SELECT CHARTS
        if charts is None:
            charts = self.charts
        last_shown = None
        for k, c in self.charts.items():
            c.plotItem.vb.state["isTopLevel"] = False
            try:
                c.plotItem.vb.sigMouseDragged.disconnect()
            except (TypeError, RuntimeError):
                pass
            try:
                c.plotItem.vb.sigMouseWheel.disconnect()
            except (TypeError, RuntimeError):
                pass
            try:
                c.plotItem.vb.sigHistoryChanged.disconnect()
            except (TypeError, RuntimeError):
                pass
            if k in charts:
                c.show()
                last_shown = c
            else:
                c.hide()
        if last_shown is not None:
            # FROM "https://github.com/pyqtgraph/pyqtgraph/pull/2010" by herodotus77
            # propagate mouse actions to charts "hidden" behind
            for k, c in self.charts.items():
                if c is not last_shown:
                    last_shown.plotItem.vb.sigMouseDragged.connect(c.plotItem.vb.mouseDragEvent)
                    last_shown.plotItem.vb.sigMouseWheel.connect(c.plotItem.vb.wheelEvent)
                    last_shown.plotItem.vb.sigHistoryChanged.connect(c.plotItem.vb.scaleHistory)
        # MOVE LEGEND TO LAYOUT
        if self.pi.legend is not None:
            self.pi.legend.setParentItem(self.pi)
        self.update()

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
            axis = self.axis[axis_name]
            charts = [self.charts[connection] for connection in axis.charts_connections]
            if axis.orientation in ["top", "bottom"]:  # IS X AXIS
                for chart in charts:
                    vb = chart.plotItem.getViewBox()
                    vb.setXRange(*range, **kwargs)
            elif axis.orientation in ["left", "right"]:  # IS Y AXIS
                for chart in charts:
                    vb = chart.plotItem.getViewBox()
                    vb.setYRange(*range, **kwargs)

    def update(self):
        """Updates all charts' contents."""
        for axis_name, axis in self.axis.items():
            if axis.autorange:
                charts = [self.charts[connection] for connection in axis.charts_connections]
                bounds = []
                if axis.orientation in ["top", "bottom"]:  # IS X AXIS
                    for chart in charts:
                        bounds += chart.dataBounds(ViewBox.XAxis)
                    bounds = [bound for bound in bounds if bound is not None]
                    if len(bounds) > 0:
                        for chart in charts:
                            vb = chart.plotItem.getViewBox()
                            vb.setXRange(min(bounds), max(bounds))
                elif axis.orientation in ["left", "right"]:  # IS Y AXIS
                    for chart in charts:
                        bounds += chart.dataBounds(ViewBox.YAxis)
                    bounds = [bound for bound in bounds if bound is not None]
                    if len(bounds) > 0:
                        for chart in charts:
                            vb = chart.plotItem.getViewBox()
                            vb.setYRange(min(bounds), max(bounds))
        super().update()

    def enableAxisAutoRange(self, axis_name):
        """Enables autorange for the axis with given name.

        Parameters:
        axis_name (str, None):
            The name of the axis to select.
        """
        self.axis[axis_name].autorange = True

    def disableAxisAutoRange(self, axis_name):
        """Disables autorange for the axis with given name.

        Parameters:
        axis_name (str, None):
            The name of the axis to select.
        """
        self.axis[axis_name].autorange = False
