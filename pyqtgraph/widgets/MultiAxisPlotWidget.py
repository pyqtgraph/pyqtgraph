# -*- coding: utf-8 -*-
__all__ = ["MultiAxisPlotWidget"]

import weakref

from ..functions import mkBrush, mkPen
from ..graphicsItems.AxisItem import AxisItem
from ..graphicsItems.PlotDataItem import PlotDataItem
from ..graphicsItems.PlotItem.PlotItem import PlotItem
from ..Qt import QtCore
from ..widgets.PlotWidget import PlotWidget


class MultiAxisPlotWidget(PlotWidget):
    # TODO: fix autorange that ignores other viewboxis with a common axis PlotDataItems.dataBounds()
    # TODO: propagate mouse events of master viewbox, to children
    # TODO: axis specific menu options for axis, propagate to linked children

    def __init__(self, **kargs):
        """PlotWidget but with support for multi axis"""
        PlotWidget.__init__(self, **kargs)
        # plotitem shortcut
        self.pi = super().getPlotItem()
        # GraphicsScene.registerObject(self.pi)
        # default vb from plotItem
        self.vb = self.pi.getViewBox()
        # GraphicsScene.registerObject(self.vb)
        # layout shortcut
        self.layout = self.pi.layout
        # hide default axis
        for a in ["left", "bottom", "right", "top"]:
            self.pi.hideAxis(a)
        # CHARTS
        self.charts = {}

    def addAxis(self, name, position, label=None, units=None, **kwargs):
        axis = AxisItem(position, **kwargs)
        axis.setLabel(label, units)
        # FUTILE
        # GraphicsScene.registerObject(axis)
        self.axis[name] = axis
        return

    def addChart(self, name, x_axis=None, y_axis=None, set_color=False, show_grid=False, **kwargs):
        # CHART
        color = self.colors[len(self.charts)]
        # ACTUAL XY GRAPH
        chart = PlotDataItem(
            name=name,
            connect="all",
            # symbol="+",
            symbol=None,
            pen=mkPen(
                color=color,
                width=2,
                s=QtCore.Qt.SolidLine,
                # brush=brush,
                c=QtCore.Qt.RoundCap,
                j=QtCore.Qt.RoundJoin
            ),
            # brush=mkBrush(
            #     color=color,
            #     bs=QtCore.Qt.SolidPattern,
            # ),
            downsampleMethod="peak",
            autoDownsample=True,
            clipToView=True
        )
        # GraphicsScene.registerObject(chart)
        if x_axis is None and y_axis is None:
            plotitem = self.pi
        else:
            # X AXIS
            if x_axis is None:
                x_axis = "bottom"
            try:
                x = self.axis[x_axis]
            except KeyError:
                self.addAxis(x_axis, "bottom", parent=plotitem)
                x = self.axis[x_axis]
            # Y AXIS
            if y_axis is None:
                y_axis = "left"
            try:
                y = self.axis[y_axis]
            except KeyError:
                self.addAxis(y_axis, "left", parent=__import__("traceback").print_stack())
                y = self.axis[y_axis]
            # VIEW
            plotitem = PlotItem(parent=self.pi, name=name + "_pi", **kwargs)
            # GraphicsScene.registerObject(plotitem)
            # hide all plotitem axis (they vould interfere with viewbox)
            for a in ["left", "bottom", "right", "top"]:
                plotitem.hideAxis(a)
            view = plotitem.getViewBox()
            # GraphicsScene.registerObject(view)
            # # Create and place axis items
            # plotitem.axes = {}
            # # link axis to new view
            self.linkAxisToView(x, view)
            self.linkAxisToView(y, view)
            for k, pos, axis in [["top", [1, 1], y], ["bottom", [3, 1], x]]:  # , ["left", [2, 0], y], ["right", [2, 2], x]]:
                # plotitem.layout.removeItem(plotitem.layout.itemAt(*pos)) # DO NOT USE, WILL MAKE AXIS UNMATCHED TO DATA, no you can't addthe new ones after it doesn't work for some reason
                plotitem.axes[k] = {"item": axis, "pos": pos}
            # fix parent legend not showing child charts
            plotitem.legend = self.pi.legend
            # resize plotitem according to the master one
            # resizing it's view doesn't work for some reason
            self.vb.sigResized.connect(lambda vb: plotitem.setGeometry(vb.sceneBoundingRect()))
        if set_color:
            # match y axis color
            y.setPen(mkPen(color=color))
        if show_grid is not False:
            if show_grid is True:
                x.setGrid(int(0.3 * 255))
                y.setGrid(int(0.3 * 255))
            else:
                if "x" in show_grid and show_grid["x"] is not False:
                    x.setGrid(int(show_grid["x"] * 255))
                if "y" in show_grid and show_grid["y"] is not False:
                    y.setGrid(int(show_grid["y"] * 255))
        plotitem.addItem(chart)
        chart.plotItem = plotitem
        self.charts[name] = chart
        self.parsed_data[name] = []
        # any(GraphicsScene.registerObject(i) for i in plotitem.items)
        # any(GraphicsScene.registerObject(i) for i in plotitem.dataItems)
        # GraphicsScene.registerObject(view.childGroup)
        return plotitem

    def linkAxisToView(self, axis, view):
        # # AUTORANGES BUT AXIS DOESNT MATCH
        # # link plotitemaxis to it's view or the view responsible for that axis
        # # if axis._linkedView is None:
        # #     axis.linkToView(view)
        # # else:
        # #     view.setXLink(axis_view)
        # # NO AUTORANGES BUT AXIS MATCHES
        # # set axis main view link if not assigned
        # # FROM AxisItem.linkToView
        if axis.linkedView() is None:
            axis._linkedView = weakref.ref(view)
        # connect view changes to axis changes
        if axis.orientation in ["right", "left"]:
            view.sigYRangeChanged.connect(axis.linkedViewChanged)
        elif axis.orientation in ["top", "bottom"]:
            view.sigXRangeChanged.connect(axis.linkedViewChanged)
        view.sigResized.connect(axis.linkedViewChanged)
        # add to AxisItem.linkToView FROM ViewBox.linkView
        axis_view = axis.linkedView()
        if axis_view is not view:
            if axis.orientation in ["right", "left"]:
                # connect axis main view changes to view
                view.state["linkedViews"][view.YAxis] = weakref.ref(axis_view)
                # connect axis main view changes to view
                axis_view.sigYRangeChanged.connect(view.linkedYChanged)
                view.enableAutoRange(view.YAxis, False)  # axis_view.autoRangeEnabled()[axis_view.YAxis])
                view.linkedYChanged()
            elif axis.orientation in ["top", "bottom"]:
                # connect axis main view changes to view
                view.state["linkedViews"][view.XAxis] = weakref.ref(axis_view)
                # connect axis main view changes to view
                axis_view.sigXRangeChanged.connect(view.linkedXChanged)
                view.enableAutoRange(view.XAxis, False)  # axis_view.autoRangeEnabled()[axis_view.XAxis])
                view.linkedXChanged()
            axis_view.sigResized.connect(view.linkedYChanged)
        view.sigStateChanged.emit(view)

    def makeLayout(self, axis=None, charts=None):
        # CLEAR LAYOUT FAILS TODO: FIX
        while self.layout.count() > 0:
            # print(self.layout.count())
            item = self.layout.itemAt(0)
            self.layout.removeItem(item)
            item.hide()
        # self.pi.setLayout(self.layout)
        # self.pi.layoutChanged.emit()
        # clear plotItem
        self.pi.clear()
        # SELECT AND ASSEMBLE AXIS
        if axis is None:
            axis = list(self.axis)
        lo = {
            "left": [],
            "right": [],
            "top": [],
            "bottom": []
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
        for y, a in enumerate(lo["top"] + [None] + lo["bottom"]):
            if a is not None:
                a.show()
                self.layout.addItem(a, y + 1, vx)
        # SELECT CHARTS
        if charts is None:
            charts = list(self.charts)
        for k, c in self.charts.items():
            if k in charts:
                c.show()
            else:
                c.hide()
        # MOVE LEGEND TO LAYOUT
        if self.pi.legend is not None:
            self.pi.legend.setParentItem(self.pi)
            # GraphicsScene.registerObject(self.pi.legend)
        # SET LAYOUT STYLE
        for x in range(self.layout.columnCount()):
            if x != vx:
                self.layout.setColumnPreferredWidth(x, 0)
                self.layout.setColumnMinimumWidth(x, 0)
                self.layout.setColumnSpacing(x, 0)
                self.layout.setColumnStretchFactor(x, 1)
            else:
                self.layout.setColumnStretchFactor(x, 100)
        for y in range(self.layout.rowCount()):
            if y != vy:
                self.layout.setRowPreferredHeight(y, 0)
                self.layout.setRowMinimumHeight(y, 0)
                self.layout.setRowSpacing(y, 0)
                self.layout.setRowStretchFactor(y, 1)
            else:
                self.layout.setRowStretchFactor(vy, 100)

    def clean(self):
        # CLEAR PLOTS
        for p in self.charts.values():
            p.clear()

    def getPlotItem(self, name=None):
        if name is None:
            return self.pi
        else:
            return self.charts[name].plotItem

    def setAxisRange(self, axis, range=None, **kwargs):
        a = self.axis[axis]
        vb = a.linkedView()
        if range is None or len(range) == 0:
            # AUTORANGE
            r = None
        elif len(range) == 1:
            # ZERO TO R
            r = [min(0, *range), max(0, *range)]
        elif len(range) == 2:
            # SET GIVEN RANGE
            r = [min(*range), max(*range)]
        else:
            raise AttributeError("bad range")
        if a.orientation in ["top", "bottom"]:
            # IS X AXIS
            if r is None:
                vb.enableAutoRange(axis=vb.XAxis, **kwargs)
            else:
                vb.setXRange(*r, **kwargs)
        elif a.orientation in ["left", "right"]:
            # IS Y AXIS
            if r is None:
                vb.enableAutoRange(axis=vb.YAxis, **kwargs)
            else:
                vb.setYRange(*r, **kwargs)
