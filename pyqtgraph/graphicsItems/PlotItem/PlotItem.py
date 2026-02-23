import collections.abc
import os
import warnings
import weakref

from typing import Iterable

import numpy as np

from ... import functions as fn
from ... import icons
from ...Qt import QtCore, QtWidgets
from ...WidgetGroup import WidgetGroup
from ...widgets.FileDialog import FileDialog
from ..AxisItem import AxisItem
from ..ButtonItem import ButtonItem
from ..GraphicsWidget import GraphicsWidget
from ..InfiniteLine import InfiniteLine
from ..LabelItem import LabelItem
from ..LegendItem import LegendItem
from ..PlotCurveItem import PlotCurveItem
from ..PlotDataItem import PlotDataItem
from ..ViewBox import ViewBox

translate = QtCore.QCoreApplication.translate

from . import plotConfigTemplate_generic as ui_template

__all__ = ['PlotItem']


class PlotItem(GraphicsWidget):
    """
    GraphicsWidget implementing a standard 2D plotting area with axes.

    **Bases:** :class:`GraphicsWidget <pyqtgraph.GraphicsWidget>`
    
    This class provides the ViewBox-plus-axes that appear when using
    :func:`pg.plot() <pyqtgraph.plot>`, :class:`PlotWidget <pyqtgraph.PlotWidget>`,
    and :meth :`GraphicsLayout.addPlot() <pyqtgraph.GraphicsLayout.addPlot>`.

    It's main functionality is:

      - Manage placement of ViewBox, AxisItems, and LabelItems
      - Create and manage a list of PlotDataItems displayed inside the ViewBox
      - Implement a context menu with commonly used display and analysis options

    Use :func:`plot() <pyqtgraph.PlotItem.plot>` to create a new PlotDataItem and
    add it to the view. Use :func:`addItem() <pyqtgraph.PlotItem.addItem>` to
    add any QGraphicsItem to the view.
    
    This class wraps several methods from its internal ViewBox:
      - :func:`setXRange <pyqtgraph.ViewBox.setXRange>`
      - :func:`setYRange <pyqtgraph.ViewBox.setYRange>`
      - :func:`setRange <pyqtgraph.ViewBox.setRange>`
      - :func:`autoRange <pyqtgraph.ViewBox.autoRange>`
      - :func:`setDefaultPadding <pyqtgraph.ViewBox.setDefaultPadding>`
      - :func:`setXLink <pyqtgraph.ViewBox.setXLink>`
      - :func:`setYLink <pyqtgraph.ViewBox.setYLink>`
      - :func:`setAutoPan <pyqtgraph.ViewBox.setAutoPan>`
      - :func:`setAutoVisible <pyqtgraph.ViewBox.setAutoVisible>`
      - :func:`setLimits <pyqtgraph.ViewBox.setLimits>`
      - :func:`viewRect <pyqtgraph.ViewBox.viewRect>`
      - :func:`viewRange <pyqtgraph.ViewBox.viewRange>`
      - :func:`setMouseEnabled <pyqtgraph.ViewBox.setMouseEnabled>`
      - :func:`enableAutoRange <pyqtgraph.ViewBox.enableAutoRange>`
      - :func:`disableAutoRange <pyqtgraph.ViewBox.disableAutoRange>`
      - :func:`setAspectLocked <pyqtgraph.ViewBox.setAspectLocked>`
      - :func:`invertY <pyqtgraph.ViewBox.invertY>`
      - :func:`invertX <pyqtgraph.ViewBox.invertX>`
      - :func:`register <pyqtgraph.ViewBox.register>`
      - :func:`unregister <pyqtgraph.ViewBox.unregister>`
    
    The ViewBox itself can be accessed by calling :func:`getViewBox() <pyqtgraph.PlotItem.getViewBox>` 
    
    Parameters
    ----------
    parent : QObject or None, default None
        Parent :class:`QObject` assign to the :class:`~pyqtgraph.PlotItem`.
    name : str or None, default None
        Register the value for this view so that others may link to it.
    labels :  dict of str or None, default None
        A dictionary specifying the axis labels to display

        .. code-block:: python
                   
            {'left': (args), 'bottom': (args), ...}
        
        The name of each axis and the corresponding arguments are passed to 
        :meth:`PlotItem.setLabel() <pyqtgraph.PlotItem.setLabel>`
        Optionally, PlotItem my also be initialized with the keyword arguments left,
        right, top, or bottom to achieve the same effect.
    title : str
        Text to set the title of the PlotItem to.
    viewBox : :class:`~pyqtgraph.ViewBox` or None, default None
        Have the PlotItem use the provided :class:`~pyqtgraph.ViewBox`. If not
        specified, the PlotItem will be constructed with this as its ViewBox.
    axisItems : dict of {'left', 'bottom', 'right', 'top } and AxisItem or None
        Pass pre-constructed :class:`~pyqtgraph.AxisItem` objects to be used for the
        `left`, `bottom`, `right` or `top` axis.  Default is None.
    enableMenu : bool
        Toggle the enabling or disabling of the right-click context menu.
    **kwargs : dict, optional
        Any extra keyword arguments are passed to
        :func:`PlotItem.plot() <pyqtgraph.PlotItem.plot>`.
    
    Signals
    -------
    sigYRangeChanged : Signal
        Signal is emitted when the range on the y-axis changes. Signal contains a
        reference to the :class:`~pyqtgraph.ViewBox`, and tuple of (xmin, xmax).
    sigXRangeChanged : Signal
        Signal is emitted when the range on the x-axis changes. Signal contains a
        reference to the :class:`~pyqtgraph.ViewBox`, and tuple of (ymin, ymax).
    sigRangeChanged : Signal
        Signal is emitted when the range on either x or y-axis changes. Signal contains
        a reference to the :class:`~pyqtgraph.ViewBox`, a list of lists of the form,
        ``[[x_min, x_max], [y_min, y_max]]`` and a list of two booleans, indicating if
        the range on which axis has changed of the form 
        ``[x_range_changed, y_range_changed]``.
    """
    
    sigRangeChanged = QtCore.Signal(object, object)    ## Emitted when the ViewBox range has changed
    sigYRangeChanged = QtCore.Signal(object, object)   ## Emitted when the ViewBox Y range has changed
    sigXRangeChanged = QtCore.Signal(object, object)   ## Emitted when the ViewBox X range has changed
    sigRangeChangedManually = QtCore.Signal(object)    ## Emitted when the ViewBox range is changed via user interaction
        
    lastFileDir = None
    
    def __init__(
        self,
        parent : QtCore.QObject | None = None,
        name: str | None = None,
        labels: dict[str, str] | None = None,
        title: str | None = None,
        viewBox: ViewBox | None = None,
        axisItems: dict[str, AxisItem] | None = None,
        enableMenu: bool = True,
        **kwargs
    ):  
        super().__init__(parent)

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )

        ## Set up control buttons
        self.autoBtn = ButtonItem(icons.getGraphPixmap('auto'), 14, self)
        self.autoBtn.mode = 'auto'
        self.autoBtn.clicked.connect(self.autoBtnClicked)
        self.buttonsHidden = False  # has the user has requested buttons to be hidden?
        self.mouseHovering = False

        self.layout = QtWidgets.QGraphicsGridLayout()
        self.layout.setContentsMargins(1,1,1,1)
        self.setLayout(self.layout)
        self.layout.setHorizontalSpacing(0)
        self.layout.setVerticalSpacing(0)

        if viewBox is None:
            viewBox = ViewBox(parent=self, enableMenu=enableMenu)
        self.vb = viewBox
        self.vb.sigStateChanged.connect(self.viewStateChanged)

        # Enable or disable plotItem menu
        self.setMenuEnabled(enableMenu, None)

        if name is not None:
            self.vb.register(name)
        self.vb.sigRangeChanged.connect(self.sigRangeChanged)
        self.vb.sigXRangeChanged.connect(self.sigXRangeChanged)
        self.vb.sigYRangeChanged.connect(self.sigYRangeChanged)
        self.vb.sigRangeChangedManually.connect(self.sigRangeChangedManually)

        self.layout.addItem(self.vb, 2, 1)
        self.alpha = 1.0
        self.autoAlpha = True
        self.spectrumMode = False

        self.legend = None

        # Initialize axis items
        self.axes = {}
        self.setAxisItems(axisItems)

        self.titleLabel = LabelItem('', size='11pt', parent=self)
        self.layout.addItem(self.titleLabel, 0, 1)
        self.setTitle(None)  ## hide

        for i in range(4):
            self.layout.setRowPreferredHeight(i, 0)
            self.layout.setRowMinimumHeight(i, 0)
            self.layout.setRowSpacing(i, 0)
            self.layout.setRowStretchFactor(i, 1)

        for i in range(3):
            self.layout.setColumnPreferredWidth(i, 0)
            self.layout.setColumnMinimumWidth(i, 0)
            self.layout.setColumnSpacing(i, 0)
            self.layout.setColumnStretchFactor(i, 1)
        self.layout.setRowStretchFactor(2, 100)
        self.layout.setColumnStretchFactor(1, 100)


        self.items = []
        self.curves = []
        self.itemMeta = weakref.WeakKeyDictionary()
        self.dataItems = []
        self.paramList = {}
        self.avgCurves = {}
        # Change these properties to adjust the appearance of the averaged curve:
        self.avgPen = fn.mkPen((0, 200, 0))
        self.avgShadowPen = fn.mkPen((0, 0, 0), width=4)

        ### Set up context menu

        w = QtWidgets.QWidget()
        self.ctrl = c = ui_template.Ui_Form()
        c.setupUi(w)

        menuItems = [
            (translate("PlotItem", 'Transforms'), c.transformGroup),
            (translate("PlotItem", 'Downsample'), c.decimateGroup),
            (translate("PlotItem", 'Average'), c.averageGroup),
            (translate("PlotItem", 'Alpha'), c.alphaGroup),
            (translate("PlotItem", 'Grid'), c.gridGroup),
            (translate("PlotItem", 'Points'), c.pointsGroup),
        ]

        self.ctrlMenu = QtWidgets.QMenu(translate("PlotItem", 'Plot Options'))

        for name, grp in menuItems:
            sm = self.ctrlMenu.addMenu(name)
            act = QtWidgets.QWidgetAction(self)
            act.setDefaultWidget(grp)
            sm.addAction(act)

        self.stateGroup = WidgetGroup()
        for name, w in menuItems:
            self.stateGroup.autoAdd(w)

        self.fileDialog = None

        c.alphaGroup.toggled.connect(self.updateAlpha)
        c.alphaSlider.valueChanged.connect(self.updateAlpha)
        c.autoAlphaCheck.toggled.connect(self.updateAlpha)

        c.xGridCheck.toggled.connect(self.updateGrid)
        c.yGridCheck.toggled.connect(self.updateGrid)
        c.gridAlphaSlider.valueChanged.connect(self.updateGrid)

        c.fftCheck.toggled.connect(self.updateSpectrumMode)
        c.subtractMeanCheck.toggled.connect(self.updateSubtractMeanMode)
        c.logXCheck.toggled.connect(self.updateLogMode)
        c.logYCheck.toggled.connect(self.updateLogMode)
        c.derivativeCheck.toggled.connect(self.updateDerivativeMode)
        c.phasemapCheck.toggled.connect(self.updatePhasemapMode)

        c.downsampleSpin.valueChanged.connect(self.updateDownsampling)
        c.downsampleCheck.toggled.connect(self.updateDownsampling)
        c.autoDownsampleCheck.toggled.connect(self.updateDownsampling)
        c.subsampleRadio.toggled.connect(self.updateDownsampling)
        c.meanRadio.toggled.connect(self.updateDownsampling)
        c.clipToViewCheck.toggled.connect(self.updateDownsampling)

        self.ctrl.avgParamList.itemClicked.connect(self.avgParamListClicked)
        self.ctrl.averageGroup.toggled.connect(self.avgToggled)

        self.ctrl.maxTracesCheck.toggled.connect(self._handle_max_traces_toggle)
        self.ctrl.forgetTracesCheck.toggled.connect(self.updateDecimation)
        self.ctrl.maxTracesSpin.valueChanged.connect(self.updateDecimation)

        if labels is None:
            labels = {}
        for label in list(self.axes.keys()):
            if label in kwargs:
                labels[label] = kwargs[label]
                del kwargs[label]
        for k in labels:
            if isinstance(labels[k], str):
                labels[k] = (labels[k],)
            self.setLabel(k, *labels[k])

        if title is not None:
            self.setTitle(title)

        if kwargs:
            self.plot(**kwargs)        
        
    def implements(self, interface=None):
        return interface in ['ViewBoxWrapper']

    def getViewBox(self) -> ViewBox:
        """
        Return the embedded :class:`~pyqtgraph.ViewBox`.

        Returns
        -------
        ViewBox 
            The :class:`ViewBox <pyqtgraph.ViewBox>` that this PlotItem uses.
        """
        return self.vb
    
    # Wrap a few methods from viewBox. 
    # Important: don't use settattr(m, getattr(self.vb, m)) as we'd be leaving the
    # viewbox alive because we had a reference to an instance method (creating wrapper
    # methods at runtime instead).

    # Note: If you update this list, please update the class docstring as well
    for m in ['setXRange', 'setYRange', 'setXLink', 'setYLink', 'setAutoPan',
              'setAutoVisible', 'setDefaultPadding', 'setRange', 'autoRange',
              'viewRect', 'viewRange', 'setMouseEnabled', 'setLimits',
              'enableAutoRange', 'disableAutoRange', 'setAspectLocked', 'invertY',
              'invertX', 'register', 'unregister']:
                
        def _create_method(name):
            def method(self, *args, **kwargs):
                return getattr(self.vb, name)(*args, **kwargs)
            method.__name__ = name
            return method
        
        locals()[m] = _create_method(m)
        
    del _create_method
    
    def setAxisItems(self, axisItems: 'dict[str, AxisItem] | None' = None) -> None:
        """
        Place axis items and initializes non-existing axis items.

        Parameters
        ----------
        axisItems : dict of str and AxisItem or None
            Optional dictionary instructing the PlotItem to use pre-constructed items
            for its axes. The dict keys must be axis names
            {'left', 'bottom', 'right', 'top'} and the values must be instances of
            :class:`~pyqtgraph.AxisItem`.
        """
        
        if axisItems is None:
            axisItems = {}
        
        # Array containing visible axis items
        # Also containing potentially hidden axes, but they are not touched so it does
        # not matter
        visibleAxes = ['left', 'bottom']
        # Note that it does not matter that this adds
        # some values to visibleAxes a second time
        visibleAxes.extend(axisItems.keys()) 
        
        for k, pos in (('top', (1,1)), ('bottom', (3,1)), ('left', (2,0)), ('right', (2,2))):
            if k in self.axes:
                if k not in axisItems:
                    continue # Nothing to do here
                
                # Remove old axis
                oldAxis = self.axes[k]['item']
                self.layout.removeItem(oldAxis)
                if oldAxis.scene() is not None:
                    oldAxis.scene().removeItem(oldAxis)
                oldAxis.unlinkFromView()
            
            # Create new axis
            if k in axisItems:
                axis = axisItems[k]
                if (
                    axis.scene() is not None 
                    and (k not in self.axes or axis != self.axes[k]["item"])
                ):
                    raise RuntimeError(
                        "Can't add an axis to multiple plots. Shared axes "
                        "can be achieved with multiple AxisItem instances "
                        "and set[X/Y]Link."
                    )
            else:
                axis = AxisItem(orientation=k, parent=self)
            
            # Set up new axis
            axis.linkToView(self.vb)
            self.axes[k] = {'item': axis, 'pos': pos}
            self.layout.addItem(axis, *pos)
            # place axis above images at z=0, items that want to draw over the axes 
            # should be placed at z>=1
            axis.setZValue(0.5) 
            axis.setFlag(
                QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemNegativeZStacksBehindParent
            )           
            axisVisible = k in visibleAxes
            self.showAxis(k, axisVisible)
        
    def setLogMode(self, x=None, y=None):
        """
        Set log scaling for `x` and/or `y` axes.

        This informs PlotDataItems to transform logarithmically and switches the axes to
        use log ticking.

        Parameters
        ----------
        x : bool or None
            If ``True``, set the x-axis to log mode. A value of ``None`` is ignored.
        y : bool or None
            If ``True``, set the y-axis to log mode. A value of ``None`` is ignored.
        
        Notes
        -----
        No other items, in the scene will be affected by this; there is (currently) 
        no generic way to redisplay a :class:`~pyqtgraph.GraphicsItem` with log
        coordinates.
        """
        if x is not None:
            self.ctrl.logXCheck.setChecked(x)
        if y is not None:
            self.ctrl.logYCheck.setChecked(y)
        
    def showGrid(self, x=None, y=None, alpha=None):
        """
        Show or hide the grid for either axis.

        Parameters
        ----------
        x : bool or None
            Show the X grid, a value of ``None`` is ignored.
        y : bool or None
            Show the Y grid, a value of ``None`` is ignored.
        alpha : float or None
            Opacity of the grid, float values need to be between [0.0, 1.0].
        """
        if x is None and y is None and alpha is None:
            # prevent people getting confused if they just call showGrid()
            raise ValueError("Must specify at least one of x, y, or alpha.")
        
        if x is not None:
            self.ctrl.xGridCheck.setChecked(x)
        if y is not None:
            self.ctrl.yGridCheck.setChecked(y)
        if alpha is not None:
            v = fn.clip_scalar(alpha, 0, 1) * self.ctrl.gridAlphaSlider.maximum() # slider range 0 to 255
            self.ctrl.gridAlphaSlider.setValue( int(v) )
        
    def close(self):
        ## Most of this crap is needed to avoid PySide trouble. 
        ## The problem seems to be whenever scene.clear() leads to deletion of widgets (either through proxies or qgraphicswidgets)
        ## the solution is to manually remove all widgets before scene.clear() is called
        if self.ctrlMenu is None: ## already shut down
            return
        self.ctrlMenu.setParent(None)
        self.ctrlMenu = None
        
        self.autoBtn.setParent(None)
        self.autoBtn = None
        
        for k in self.axes:
            i = self.axes[k]['item']
            i.close()
            
        self.axes = None
        self.scene().removeItem(self.vb)
        self.vb = None
        
    def registerPlot(self, name):   ## for backward compatibility
        self.vb.register(name)
        
    @QtCore.Slot(bool)
    @QtCore.Slot(int)
    def updateGrid(self, *args):
        alpha = self.ctrl.gridAlphaSlider.value()
        x = alpha if self.ctrl.xGridCheck.isChecked() else False
        y = alpha if self.ctrl.yGridCheck.isChecked() else False
        self.getAxis('top').setGrid(x)
        self.getAxis('bottom').setGrid(x)
        self.getAxis('left').setGrid(y)
        self.getAxis('right').setGrid(y)

    def viewGeometry(self) -> QtCore.QRectF:
        """
        Return the screen geometry of the viewbox.

        Returns
        -------
        QRectF
            The QRectF instance that contains the view area.
        """
        v = self.scene().views()[0]
        b = self.vb.mapRectToScene(self.vb.boundingRect())
        wr = v.mapFromScene(b).boundingRect()
        pos = v.mapToGlobal(v.pos())
        wr.adjust(pos.x(), pos.y(), pos.x(), pos.y())
        return wr

    @QtCore.Slot(bool)
    def avgToggled(self, b: bool):
        if b:
            self.recomputeAverages()
        for k in self.avgCurves:
            self.avgCurves[k][1].setVisible(b)
        
    @QtCore.Slot(QtWidgets.QListWidgetItem)
    def avgParamListClicked(self, item):
        name = str(item.text())
        self.paramList[name] = (item.checkState() == QtCore.Qt.CheckState.Checked)
        self.recomputeAverages()
        
    def recomputeAverages(self):
        if not self.ctrl.averageGroup.isChecked():
            return
        for k in self.avgCurves:
            self.removeItem(self.avgCurves[k][1])
        self.avgCurves = {}
        for c in self.curves:
            self.addAvgCurve(c)
        self.replot()
        
    def addAvgCurve(self, curve):
        ## Add a single curve into the pool of curves averaged together

        ## If there are plot parameters, then we need to determine which to average together.
        remKeys = []
        addKeys = []
        if self.ctrl.avgParamList.count() > 0:

            # First determine the key of the curve to which this new data should be averaged
            for i in range(self.ctrl.avgParamList.count()):
                item = self.ctrl.avgParamList.item(i)
                if item.checkState() == QtCore.Qt.CheckState.Checked:
                    remKeys.append(str(item.text()))
                else:
                    addKeys.append(str(item.text()))

            if not remKeys:
                # In this case,there would be 1 avg plot for each data plot; not useful
                return

        p = self.itemMeta.get(curve,{}).copy()
        for k in p:
            if type(k) is tuple:
                p['.'.join(k)] = p[k]
                del p[k]
        for rk in remKeys:
            if rk in p:
                del p[rk]
        for ak in addKeys:
            if ak not in p:
                p[ak] = None
        key = tuple(p.items())

        ### Create a new curve if needed
        if key not in self.avgCurves:
            plot = PlotDataItem()
            plot.setPen( self.avgPen )
            plot.setShadowPen(  self.avgShadowPen )
            plot.setAlpha(1.0, False)
            plot.setZValue(100)
            self.addItem(plot, skipAverage=True)
            self.avgCurves[key] = [0, plot]
        self.avgCurves[key][0] += 1
        (n, plot) = self.avgCurves[key]

        ### Average data together
        (x, y) = curve.getData()
        stepMode = curve.opts['stepMode']
        if plot.yData is not None and y.shape == plot.yData.shape:
            # note that if shapes do not match, then the average resets.
            newData = plot.yData * (n-1) / float(n) + y * 1.0 / float(n)
            plot.setData(plot.xData, newData, stepMode=stepMode)
        else:
            plot.setData(x, y, stepMode=stepMode)
        
    @QtCore.Slot()
    def autoBtnClicked(self):
        if self.autoBtn.mode == 'auto':
            self.enableAutoRange()
            self.autoBtn.hide()
            self.sigRangeChangedManually.emit(self.vb.mouseEnabled()[:])
        else:
            self.disableAutoRange()
            
    @QtCore.Slot()
    def viewStateChanged(self):
        self.updateButtons()

    def addItem(self, item, *args, **kwargs):
        """
        Add a :class:`~pyqtgraph.GraphicsItem` to the :class:`~pyqtgraph.ViewBox`.
        
        If the item has plot data (:class:`PlotDataItem <pyqtgraph.PlotDataItem>` , 
        :class:`~pyqtgraph.PlotCurveItem`, :class:`~pyqtgraph.ScatterPlotItem` ), it may
        be included in analysis performed by the PlotItem.

        Parameters
        ----------
        item : GraphicsItem
            Item to add to the ViewBox.
        *args : tuple
            Arguments relayed to :meth:`~pyqtgraph.ViewBox.addItem`.
        **kwargs : dict
            Keyword arguments for adding an item.  Supported arguments include.

            ============ ===============================================================
            Property     Description
            ============ ===============================================================
            ignoreBounds ``bool`` - Keyword argument relayed to 
                         :meth:`~pyqtgraph.ViewBox.addItem`.
            
            skipAverage  ``bool`` - If ``True``, do not use curve to compute average
                         curves.
            ============ ===============================================================

        See Also
        --------
        :meth:`~pyqtgraph.ViewBox.addItem`
            See method for supported arguments to be passed.
        """
        if item in self.items:
            warnings.warn('Item already added to PlotItem, ignoring.')
            return
        self.items.append(item)
        vbargs = {}
        if 'ignoreBounds' in kwargs:
            vbargs['ignoreBounds'] = kwargs['ignoreBounds']
        self.vb.addItem(item, *args, **vbargs)
        name = None
        if hasattr(item, 'implements') and item.implements('plotData'):
            name = item.name()
            self.dataItems.append(item)            
            params = kwargs.get('params', {})
            self.itemMeta[item] = params
            self.curves.append(item)
        
        # Toggle log mode if item implements setLogMode and selected in the context menu
        if hasattr(item, 'setLogMode'):
            item.setLogMode(
                self.ctrl.logXCheck.isChecked(),
                self.ctrl.logYCheck.isChecked()
            )
            
        if isinstance(item, PlotDataItem):
            ## configure curve for this plot
            (alpha, auto) = self.alphaState()
            item.setAlpha(alpha, auto)
            item.setSubtractMeanMode(self.ctrl.subtractMeanCheck.isChecked())
            item.setFftMode(self.ctrl.fftCheck.isChecked())
            item.setDownsampling(*self.downsampleMode())
            item.setClipToView(self.clipToViewMode())
            
            ## Hide older plots if needed
            self.updateDecimation()
            
            ## Add to average if needed
            self.updateParamList()
            if self.ctrl.averageGroup.isChecked() and 'skipAverage' not in kwargs:
                self.addAvgCurve(item)

        # conditionally add item to the legend
        if name is not None and hasattr(self, 'legend') and self.legend is not None:
            self.legend.addItem(item, name=name)            

    def listDataItems(self):
        """
        Return a list of all data items current plotted.

        Returns
        -------
        list of PlotCurveItem, ScatterPlotItem or PlotDataItem
            A copy of a list of the data items.
        """
        return self.dataItems[:]

    def addLine(self, x=None, y=None, z=None, **kwargs):
        """
        Create a new :class:`~pyqtgraph.InfiniteLine` and add it to the plot.

        Parameters
        ----------
        x : float or None
            Position in the x-axis to draw the line. If specified, the line will be 
            vertical. Default ``None``.
        y : float or None
            Position in the y-axis to draw the line. If specified, the line will be
            horizontal. Default ``None``.
        z : int or None
            Z value to set the line to. This is used to determine which items are on top
            of the other.  See :meth:`~QtWidgets.QGraphicsItem.setZValue`.
        **kwargs : dict
            Keyword arguments to pass to the :class:`~pyqtgraph.InfiniteLine` instance.
        
        Returns
        -------
        InfiniteLine
            The new :class:`~pyqtgraph.InfiniteLine` added.
        """
        kwargs['pos'] = kwargs.get('pos', x if x is not None else y)
        kwargs['angle'] = kwargs.get('angle', 0 if x is None else 90)
        line = InfiniteLine(**kwargs)
        self.addItem(line)
        if z is not None:
            line.setZValue(z)
        return line

    def removeItem(self, item):
        """
        Remove an item from the plot.

        Parameters
        ----------
        item : GraphicsItem
            The item to remove.
        """
        if item not in self.items:
            return
        self.items.remove(item)
        if item in self.dataItems:
            self.dataItems.remove(item)

        self.vb.removeItem(item)

        if item in self.curves:
            self.curves.remove(item)
            self.updateDecimation()
            self.updateParamList()

        if self.legend is not None:
            self.legend.removeItem(item)

    def clear(self):
        """
        Remove all items from the PlotItem's :class:`~pyqtgraph.ViewBox`.
        """
        for i in self.items[:]:
            self.removeItem(i)
        self.avgCurves = {}
    
    def clearPlots(self):
        """
        Remove all curves from the PlotItem's :class:`~pyqtgraph.ViewBox`.
        """
        for i in self.curves[:]:
            self.removeItem(i)
        self.avgCurves = {}        
    
    def plot(self, *args, **kwargs):
        """
        Add and return a new :class:`~pyqtgraph.PlotDataItem`.

        Parameters
        ----------
        *args : tuple, optional
            Arguments that are passed to the :class:`~pyqtgraph.PlotDataItem`
            constructor.
        **kwargs : dict, optional
            Keyword arguments that are passed to the :class:`~pyqtgraph.PlotDataItem`
            constructor.  In addition, the following keyword arguments are accepted.

            =========== ================================================================
            Property    Description
            =========== ================================================================
            clear       ``bool`` - Call :meth:`~pyqtgraph.PlotItem.clear` prior to
                        creating the new plot instance.
            
            params      ``dict`` or ``None`` - Arguments to passed as the `params`
                        argument to :meth:`~pyqtgraph.PlotItem.addItem`.
            =========== ================================================================

        Returns
        -------
        PlotDataItem
            The newly created :class:`~pyqtgraph.PlotDataItem`.
        """
        clear = kwargs.get('clear', False)
        params = kwargs.get('params')
          
        if clear:
            self.clear()
            
        item = PlotDataItem(*args, **kwargs)
            
        if params is None:
            params = {}
        self.addItem(item, params=params)
        
        return item

    def addLegend(self, offset=(30, 30), **kwargs):
        """
        Add a new :class:`~pyqtgraph.LegendItem`.
         
        After the LegendItem is created, it is anchored in the internal 
        :class:`~pyqtgraph.ViewBox`. Plots added after this will be automatically 
        displayed in the legend if they are created with a 'name' argument.

        If a :class:`~pyqtgraph.LegendItem` has already been created using this method, 
        that item will be returned rather than creating a new one.

        Parameters
        ----------
        offset : tuple of int, int
            The distance to offset the LegendItem, defaults to ``(30, 30)``.
        **kwargs : dict, optional
            Keyword argument passed to the :class:`~pyqtgraph.LegendItem` constructor.
        
        Returns
        -------
        LegendItem
            The instance of :class:`~pyqtgraph.LegendItem` that was constructed.
        """
        if self.legend is None:
            self.legend = LegendItem(offset=offset, **kwargs)
            self.legend.setParentItem(self.vb)
        return self.legend
        
    def addColorBar(self, image, **kwargs):
        """
        Add a ColorBarItem and set to the provided image.
        
        A call like ``plot.addColorBar(img, colorMap='viridis')`` is a convenient
        method to assign and show a color map.

        Parameters
        ----------
        image : ImageItem or list of ImageItem
            See :meth:`~pyqtgraph.ColorBarItem.setImageItem` for details.
        **kwargs : dict, optional
            Keyword arguments passed to the :class:`~pyqtgraph.ColorBarItem`
            constructor.
        
        Returns
        -------
        ColorBarItem
            The newly created :class:`~pyqtgraph.ColorBarItem` instance.
        """
        from ..ColorBarItem import ColorBarItem  # avoid circular import
        bar = ColorBarItem(**kwargs)
        bar.setImageItem( image, insert_in=self )
        return bar

    def multiDataPlot(self, *, x=None, y=None, constKwargs=None, **kwargs):
        """
        Allow plotting multiple curves on the same plot in one call.

        Parameters
        ----------
        x, y : array_like
            Can be in the following formats:
              - {x or y} = [n1, n2, n3, ...]: The named argument iterates through
                ``n`` curves, while the unspecified argument is range(len(n)) for
                each curve.
              - x, [y1, y2, y3, ...]
              - [x1, x2, x3, ...], [y1, y2, y3, ...]
              - [x1, x2, x3, ...], y

            where ``x_n`` and ``y_n`` are ``ndarray`` data for each curve. Since
            ``x`` and ``y`` values are matched using ``zip``, unequal lengths mean
            the longer array will be truncated. Note that 2D matrices for either x
            or y are considered lists of curve data.
        constKwargs : dict, optional
            A dict of {str: value} passed to each curve during ``plot()``.
        **kwargs : dict, optional
            A dict of {str: iterable} where the str is the name of a kwarg and the
            iterable is a list of values, one for each plotted curve.
        
        Returns
        -------
        list of PlotDataItem
            Returns a list of the newly constructed :class:`~pyqtgraph.PlotDataItem`
            instances representing the new curves.
        """
        if (x is not None and not len(x)) or (y is not None and not len(y)):
            # Nothing to plot -- either x or y array will bail out early from
            # zip() below.
            return []
        def scalarOrNone(val):
            return val is None or (len(val) and np.isscalar(val[0]))

        if scalarOrNone(x) and scalarOrNone(y):
            raise ValueError(
                "If both `x` and `y` represent single curves, use `plot` instead "
                "of `multiPlot`."
            )
        curves = []
        constKwargs = constKwargs or {}
        xy: dict[str, list | None] = dict(x=x, y=y)
        for key, oppositeVal in zip(('x', 'y'), [y, x]):
            oppositeVal: Iterable | None
            val = xy[key]
            if val is None:
                # Other curve has all data, make range that supports longest chain
                val = range(max(len(curveN) for curveN in oppositeVal))
            if np.isscalar(val[0]):
                # x, [y1, y2, y3, ...] or [x1, x2, x3, ...], y
                # Repeat the single curve to match length of opposite list
                val = [val] * len(oppositeVal)
            xy[key] = val
        for ii, (xi, yi) in enumerate(zip(xy['x'], xy['y'])):
            for kk in kwargs:
                if len(kwargs[kk]) <= ii:
                    raise ValueError(
                        f"Not enough values for kwarg `{kk}`. "
                        f"Expected {ii + 1:d} (number of curves to plot), got"
                        f" {len(kwargs[kk]):d}"
                    )
            kwargs_i = {kk: kwargs[kk][ii] for kk in kwargs}
            constKwargs.update(kwargs_i)
            curves.append(self.plot(xi, yi, **constKwargs))
        return curves

    def scatterPlot(self, *args, **kwargs):
        """
        Create a :class:`~pyqtgraph.PlotDataItem` and add it to the plot.

        The :class:`~pyqtgraph.PlotDataItem` instance will render the underlying data
        in a scatter plot form.

        Parameters
        ----------
        *args : tuple, optional
            Arguments that are passed to the :class:`~pyqtgraph.PlotDataItem`
            constructor.
        **kwargs : dict, optional
            Keyword arguments that are passed to the :class:`~pyqtgraph.PlotDataItem`
            constructor.  In addition, the following keyword arguments are accepted.

            =========== ================================================================
            Property    Description
            =========== ================================================================
            pen         ``QPen`` - Sets the pen used to draw lines or symbols for the
                        plot.  Equivalent to ``symbolPen``.

            brush       ``QBrush`` - Sets the brush used to fill symbols for the plot.
                        Equivalent to ``symbolBrush``.

            size        ``float`` - Sets the symbol size for the plot. Equivalent to
                        ``symbolSize``.
            =========== ================================================================

        Returns
        -------
        PlotDataItem
            The newly created :class:`~pyqtgraph.PlotDataItem`.
        """
        if 'pen' in kwargs:
            kwargs['symbolPen'] = kwargs['pen']
        kwargs['pen'] = None
            
        if 'brush' in kwargs:
            kwargs['symbolBrush'] = kwargs['brush']
            del kwargs['brush']
            
        if 'size' in kwargs:
            kwargs['symbolSize'] = kwargs['size']
            del kwargs['size']

        return self.plot(*args, **kwargs)
                
    def replot(self):
        self.update()

    def updateParamList(self):
        self.ctrl.avgParamList.clear()
        ## Check to see that each parameter for each curve is present in the list
        for c in self.curves:
            for p in list(self.itemMeta.get(c, {}).keys()):
                if type(p) is tuple:
                    p = '.'.join(p)

                if matches := self.ctrl.avgParamList.findItems(
                    p, QtCore.Qt.MatchFlag.MatchExactly
                ):
                    i = matches[0]
                else:
                    i = QtWidgets.QListWidgetItem(p)
                    if p in self.paramList and self.paramList[p] is True:
                        i.setCheckState(QtCore.Qt.CheckState.Checked)
                    else:
                        i.setCheckState(QtCore.Qt.CheckState.Unchecked)
                    self.ctrl.avgParamList.addItem(i)
                self.paramList[p] = (i.checkState() == QtCore.Qt.CheckState.Checked)

    def writeSvg(self, fileName=None):
        """
        Write the plot content to an SVG file.

        Parameters
        ----------
        fileName : str, optional
            The name of the file to write to. If not specified, a file dialog will be
            opened.
        """
        if fileName is None:
            self._chooseFilenameDialog(handler=self.writeSvg)
            return

        fileName = str(fileName)
        PlotItem.lastFileDir = os.path.dirname(fileName)
        
        from ...exporters import SVGExporter
        ex = SVGExporter(self)
        ex.export(fileName)
        
    def writeImage(self, fileName=None):
        """
        Write the plot content to an image file.

        Parameters
        ----------
        fileName : str, optional
            The name of the file to write to. If not specified, a file dialog will be
            opened.
        """
        if fileName is None:
            self._chooseFilenameDialog(handler=self.writeImage)
            return

        from ...exporters import ImageExporter
        ex = ImageExporter(self)
        ex.export(fileName)
        
    def writeCsv(self, fileName=None):
        """
        Write the plot data to a CSV file.

        Parameters
        ----------
        fileName : str, optional
            The name of the file to write to. If not specified, a file dialog will be
            opened.
        """
        if fileName is None:
            self._chooseFilenameDialog(handler=self.writeCsv)
            return

        fileName = str(fileName)
        PlotItem.lastFileDir = os.path.dirname(fileName)
        
        data = [c.getData() for c in self.curves]
        with open(fileName, 'w') as fd:
            i = 0
            while True:
                done = True
                for d in data:
                    if i < len(d[0]):
                        fd.write('%g,%g,' % (d[0][i], d[1][i]))
                        done = False
                    else:
                        fd.write(' , ,')
                fd.write('\n')
                if done:
                    break
                i += 1

    def saveState(self):
        state = self.stateGroup.state()
        state['paramList'] = self.paramList.copy()
        state['view'] = self.vb.getState()
        return state
        
    def restoreState(self, state):
        if 'paramList' in state:
            self.paramList = state['paramList'].copy()
            
        self.stateGroup.setState(state)
        self.updateSpectrumMode()
        self.updateSubtractMeanMode()
        self.updateDownsampling()
        self.updateAlpha()
        self.updateDecimation()
        
        if 'powerSpectrumGroup' in state:
            state['fftCheck'] = state['powerSpectrumGroup']
        if 'gridGroup' in state:
            state['xGridCheck'] = state['gridGroup']
            state['yGridCheck'] = state['gridGroup']
            
        self.stateGroup.setState(state)
        self.updateParamList()
        
        if 'view' not in state:
            r = [[float(state['xMinText']), float(state['xMaxText'])], [float(state['yMinText']), float(state['yMaxText'])]]
            state['view'] = {
                'autoRange': [state['xAutoRadio'], state['yAutoRadio']],
                'linkedViews': [state['xLinkCombo'], state['yLinkCombo']],
                'targetRange': r,
                'viewRange': r,
            }
        self.vb.setState(state['view'])

    def widgetGroupInterface(self):
        return (None, PlotItem.saveState, PlotItem.restoreState)
      
    @QtCore.Slot(bool)
    def updateSpectrumMode(self, b=None):
        if b is None:
            b = self.ctrl.fftCheck.isChecked()
        for c in self.curves:
            c.setFftMode(b)
        self.enableAutoRange()
        self.recomputeAverages()

    @QtCore.Slot()
    def updateSubtractMeanMode(self):
        d = self.ctrl.subtractMeanCheck.isChecked()
        for i in self.items:
            if hasattr(i, 'setSubtractMeanMode'):
                i.setSubtractMeanMode(d)
        self.enableAutoRange()
        self.recomputeAverages()

    @QtCore.Slot()
    def updateLogMode(self):
        x = self.ctrl.logXCheck.isChecked()
        y = self.ctrl.logYCheck.isChecked()
        for i in self.items:
            if hasattr(i, 'setLogMode'):
                i.setLogMode(x,y)
        self.getAxis('bottom').setLogMode(x, y)
        self.getAxis('top').setLogMode(x, y)
        self.getAxis('left').setLogMode(x, y)
        self.getAxis('right').setLogMode(x, y)
        self.enableAutoRange()
        self.recomputeAverages()
    
    @QtCore.Slot()
    def updateDerivativeMode(self):
        d = self.ctrl.derivativeCheck.isChecked()
        for i in self.items:
            if hasattr(i, 'setDerivativeMode'):
                i.setDerivativeMode(d)
        self.enableAutoRange()
        self.recomputeAverages()

    @QtCore.Slot()
    def updatePhasemapMode(self):
        d = self.ctrl.phasemapCheck.isChecked()
        for i in self.items:
            if hasattr(i, 'setPhasemapMode'):
                i.setPhasemapMode(d)
        self.enableAutoRange()
        self.recomputeAverages()
        
        
    def setDownsampling(self, ds=None, auto=None, mode=None):
        """
        Set the downsampling mode for the PlotItem.

        Downsampling reduces the number of samples drawn to improve performance.
        The downsampling mode can be set to 'subsample', 'mean', 'peak', or None.
        If downsampling is enabled, the view will display downsampled data when zoomed out,
        but will display original data at high zoom levels.

        Parameters
        ----------
        ds : int or bool or None, optional
            The downsampling factor. If ``None``, the downsampling factor is not
            changed. If ``True``, the downsampling factor is set to the value in the
            downsampling spin box. If ``False``, downsampling is disabled. If an
            integer, the downsampling factor is set to this value. Default is ``None``.
        auto : bool or None, optional
            If ``True``, automatic downsampling is enabled. If ``False``, automatic
            downsampling is disabled. If ``None``, the automatic downsampling setting
            is not changed. Default is ``None``.
        mode : {'subsample', 'mean', 'peak'} or None, optional
            The downsampling mode. If ``None``, the downsampling mode is not changed.
            Default is ``None``.
        
        Raises
        ------
        ValueError
            Raised if the specified downsample mode is not recognized.
        """

        if ds is not None:
            if ds is False:
                self.ctrl.downsampleCheck.setChecked(False)
            elif ds is True:
                self.ctrl.downsampleCheck.setChecked(True)
            else:
                self.ctrl.downsampleCheck.setChecked(True)
                self.ctrl.downsampleSpin.setValue(ds)
                
        if auto is not None:
            if auto and ds is not False:
                self.ctrl.downsampleCheck.setChecked(True)
            self.ctrl.autoDownsampleCheck.setChecked(auto)
            
        if mode is not None:
            if mode == 'subsample':
                self.ctrl.subsampleRadio.setChecked(True)
            elif mode == 'mean':
                self.ctrl.meanRadio.setChecked(True)
            elif mode == 'peak':
                self.ctrl.peakRadio.setChecked(True)
            else:
                raise ValueError(
                    "mode argument must be 'subsample', 'mean', or 'peak'."
                )
            
    @QtCore.Slot()
    def updateDownsampling(self):
        ds, auto, method = self.downsampleMode()
        clip = self.ctrl.clipToViewCheck.isChecked()
        for c in self.curves:
            c.setDownsampling(ds, auto, method)
            c.setClipToView(clip)
        self.recomputeAverages()
        
    def downsampleMode(self) -> tuple[int, bool, str]:
        if self.ctrl.downsampleCheck.isChecked():
            ds = self.ctrl.downsampleSpin.value()
        else:
            ds = 1
            
        auto = (
            self.ctrl.downsampleCheck.isChecked() and 
            self.ctrl.autoDownsampleCheck.isChecked()
        )
            
        if self.ctrl.subsampleRadio.isChecked():
            method = 'subsample' 
        elif self.ctrl.meanRadio.isChecked():
            method = 'mean'
        elif self.ctrl.peakRadio.isChecked():
            method = 'peak'
        else:
            raise ValueError(
                "One of the method radios must be selected for: 'subsample', 'mean', "
                "'peak'."
            )
        return ds, auto, method
        
    def setClipToView(self, clip):
        """
        Set the default clip-to-view mode for new curves.

        Parameters
        ----------
        clip : bool
            If ``True``, new curves will be clipped to the view box.
        """
        self.ctrl.clipToViewCheck.setChecked(clip)
        
    def clipToViewMode(self) -> bool:
        """
        Return whether clip-to-view mode is enabled.

        Returns
        -------
        bool
            ``True`` if clip-to-view mode is enabled, ``False`` otherwise.
        """
        
        return self.ctrl.clipToViewCheck.isChecked()
    
    @QtCore.Slot(bool)
    def _handle_max_traces_toggle(self, check_state):
        if check_state:
            self.updateDecimation()
        else:
            for curve in self.curves:
                curve.show()
    
    @QtCore.Slot()
    def updateDecimation(self):
        """
        Update the number of visible curves.

        Reduce or increase number of visible curves according to value set by the 
        `Max Traces` spinner, if `Max Traces` is checked in the context menu. Destroy
        curves that are not visible if `forget traces` is checked. In most cases, this
        function is called automatically when the `Max Traces` GUI elements are
        triggered. It is also called when the state of :class:`~pyqtgraph.PlotItem` is
        updated, its state is restored, or new items added added/removed.
        
        This can cause an unexpected or conflicting state of curve visibility
        (or destruction) if curve visibilities are controlled externally. In the case of
        external control it is advised to disable the `Max Traces` checkbox (or context
        menu) to prevent unexpected curve state changes.

        :meta private:
        """
        if not self.ctrl.maxTracesCheck.isChecked():
            return
        else:
            numCurves = self.ctrl.maxTracesSpin.value()

        if self.ctrl.forgetTracesCheck.isChecked():
            for curve in self.curves[:-numCurves]:
                curve.clear()
                self.removeItem(curve)

        for i, curve in enumerate(reversed(self.curves)):
            if i < numCurves:
                curve.show()
            else:
                curve.hide()
      
    @QtCore.Slot(bool)
    @QtCore.Slot(int)
    def updateAlpha(self, *args):
        (alpha, auto) = self.alphaState()
        for c in self.curves:
            c.setAlpha(alpha**2, auto)
     
    def alphaState(self):
        enabled = self.ctrl.alphaGroup.isChecked()
        auto = self.ctrl.autoAlphaCheck.isChecked()
        alpha = float(self.ctrl.alphaSlider.value()) / self.ctrl.alphaSlider.maximum()
        if auto:
            alpha = 1.0  ## should be 1/number of overlapping plots
        if not enabled:
            auto = False
            alpha = 1.0
        return (alpha, auto)

    def pointMode(self):
        if self.ctrl.pointsGroup.isChecked():
            return None if self.ctrl.autoPointsCheck.isChecked() else True
        else:
            return False

    def resizeEvent(self, ev):
        if self.autoBtn is None:  ## already closed down
            return
        btnRect = self.mapRectFromItem(self.autoBtn, self.autoBtn.boundingRect())
        y = self.size().height() - btnRect.height()
        self.autoBtn.setPos(0, y)
    
    def getMenu(self):
        return self.ctrlMenu
    
    def getContextMenus(self, event):
        ## called when another item is displaying its context menu; we get to add extras to the end of the menu.
        if self.menuEnabled():
            return self.ctrlMenu
        else:
            return None
    
    def setMenuEnabled(self, enableMenu=True, enableViewBoxMenu='same'):
        """
        Enable or disable the context menu for this PlotItem.

        By default, the ViewBox's context menu will also be affected. Use 
        ``enableViewBoxMenu=None`` to leave the ViewBox unchanged.

        Parameters
        ----------
        enableMenu : bool, optional
            Whether to enable or disable the context menu, by default True.
        enableViewBoxMenu : str or bool, optional
            Whether to enable or disable the ViewBox context menu.
            If 'same', the ViewBox menu will be enabled or disabled as *enableMenu*.
            If ``True`` or ``False``, the ViewBox menu will be set accordingly.
            If ``None``, the ViewBox menu will not be changed.
            By default 'same'.
        """
        self._menuEnabled = enableMenu
        if enableViewBoxMenu is None:
            return
        if enableViewBoxMenu == 'same':
            enableViewBoxMenu = enableMenu
        self.vb.setMenuEnabled(enableViewBoxMenu)
    
    def menuEnabled(self):
        """
        Return whether the context menu is enabled.

        Returns
        -------
        bool
            ``True`` if the context menu is enabled, ``False`` otherwise.
        """
        return self._menuEnabled

    def setContextMenuActionVisible(self, name : str, visible : bool) -> None:
        """
        Change the context menu action visibility.

        Parameters
        ----------
        name : {'Transforms', 'Downsample', 'Average', 'Alpha', 'Grid', 'Points'}
            Action name.
        visible : bool
            Determines if action will be display. ``True`` action is visible, ``False``
            action is invisible.
        """
        translated_name = translate("PlotItem", name)
        for action in self.ctrlMenu.actions():
            if action.text() == translated_name:
                action.setVisible(visible)
                break
    
    def hoverEvent(self, ev):
        if ev.enter:
            self.mouseHovering = True
        if ev.exit:
            self.mouseHovering = False
            
        self.updateButtons()

    def getLabel(self, key):
        pass
        
    def _checkScaleKey(self, key):
        if key not in self.axes:
            raise KeyError(
                f"Scale '{key}' not found. Scales are: {list(self.axes.keys())}"
            )
        
    def getScale(self, key):
        return self.getAxis(key)
        
    def getAxis(self, name):
        """
        Return the specified AxisItem.

        Parameters
        ----------
        name : {'left', 'bottom', 'right', 'top'}
            The name of the axis to return.

        Returns
        -------
        AxisItem
            The :class:`~pyqtgraph.AxisItem`.

        Raises
        ------
        KeyError
            If the specified axis is not present.
        """
        self._checkScaleKey(name)
        return self.axes[name]['item']
        
    def setLabel(self, axis: str, *args, **kwargs):
        """
        Set the label for an axis.
        
        Basic HTML is allowed. See :func:`AxisItem.setLabel` for formatting options.
        
        Parameters
        ----------
        axis : {'left', 'bottom', 'right', 'top'}
            Specify which :class:`~pyqtgraph.AxisItem` to set the label for.
        *args : tuple, optional
            All extra arguments are passed to :meth:`~pyqtgraph.AxisItem.setLabel`.
        **kwargs : dict, optional
            Keyword arguments are passed to :meth:`~pyqtgraph.AxisItem.setLabel`.
        """
        self.getAxis(axis).setLabel(*args, **kwargs)
        self.showAxis(axis)
        
    def setLabels(self, **kwargs):
        """
        Set the axis labels of the plot.

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments are passed to  :meth:`~pyqtgraph.AxisItem.setLabel`.  The
            special keyword ``title`` can be used to set the plot title using
            :meth:`~pyqtgraph.PlotItem.setTitle`.
        """
        for k, v in kwargs.items():
            if k == 'title':
                self.setTitle(v)
            else:
                if isinstance(v, str):
                    v = (v,)
                self.setLabel(k, *v)
        
    def showLabel(self, axis, show=True):
        """
        Show or hide one of the plot's axis labels.

        Parameters
        ----------
        axis : {'left', 'bottom', 'right', 'top'}
            The axis label to show or hide.
        show : bool, optional
            Whether to show or hide the label, by default True.
        """
        self.getScale(axis).showLabel(show)

    def setTitle(self, title=None, **args):
        """
        Set the title of the plot.

        Parameters
        ----------
        title : str, optional
            The title text. If ``None``, the title will be hidden. The default is
            ``None``.
        **args : dict
            Additional keyword arguments are passed to
            :meth:`~pyqtgraph.LabelItem.setText`.
        """
        if title is None:
            self.titleLabel.setVisible(False)
            self.layout.setRowFixedHeight(0, 0)
            self.titleLabel.setMaximumHeight(0)
        else:
            self.titleLabel.setMaximumHeight(30)
            self.layout.setRowFixedHeight(0, 30)
            self.titleLabel.setVisible(True)
            self.titleLabel.setText(title, **args)

    def showAxis(self, axis: str, show: bool=True):
        """
        Show or hide an axis.

        Parameters
        ----------
        axis : {'left', 'bottom', 'right', 'top'}
            The axis to show or hide.
        show : bool, optional
            Whether to show or hide the axis, by default True.
        """
        s = self.getScale(axis)
        if show:
            s.show()
        else:
            s.hide()
            
    def hideAxis(self, axis):
        """
        Hide an axis.

        Parameters
        ----------
        axis : {'left', 'bottom', 'right', 'top'}
            The axis to hide.
        """
        self.showAxis(axis, False)
        
    def showAxes(self, selection, showValues=True, size=False):
        """ 
        Convenience method for quickly configuring axis settings.
        
        Parameters
        ----------
        selection : bool or tuple of bool 
            Determines which AxisItems will be displayed. If in tuple form, order is
            (left, top, right, bottom). A single boolean value will set all axes, so
            that ``showAxes(True)`` configures the axes to draw a frame.
        showValues : bool or tuple of bool, optional
            Determines if values will be displayed for the ticks of each axis. ``True``
            value shows values for left and bottom axis (default). ``False`` shows no
            values. If in tuple form, order is (left, top, right, bottom). ``None``
            leaves settings unchanged. If not specified, left and bottom axes will be
            drawn with values.
        size : float or tuple of float, optional
            Reserves as fixed amount of space (width for vertical axis, height for
            horizontal axis) for each axis where tick values are enabled. If only a
            single float value is given, it will be applied for both width and height.
            If ``None`` is given instead of a float value, the axis reverts to automatic
            allocation of space. If in tuple form, order is ``(width, height)``.
        """
        if selection is True: # shortcut: enable all axes, creating a frame
            selection = (True, True, True, True)
        elif selection is False: # shortcut: disable all axes
            selection = (False, False, False, False)
        if showValues is True: # shortcut: defaults arrangement with labels at left and bottom
            showValues = (True, False, False, True)
        elif showValues is False: # shortcut: disable all labels
            showValues = (False, False, False, False)
        elif showValues is None: # leave labelling untouched
            showValues = (None, None, None, None)
        if size is not False and not isinstance(size, collections.abc.Sized):
            size = (size, size) # make sure that size is either False or a full set of (width, height)

        all_axes = ('left','top','right','bottom')
        for show_axis, show_value, axis_key in zip(selection, showValues, all_axes):
            if show_axis is not None:
                if show_axis: self.showAxis(axis_key)
                else        : self.hideAxis(axis_key)

            if show_value is not None:
                ax = self.getAxis(axis_key)
                ax.setStyle(showValues=show_value)
                if size is not False: # size adjustment is requested
                    if axis_key in ('left','right'):
                        if show_value: ax.setWidth(size[0])
                        else         : ax.setWidth( None )
                    elif axis_key in ('top', 'bottom'):
                        if show_value: ax.setHeight(size[1])
                        else         : ax.setHeight( None )
  
    def hideButtons(self):
        """
        Hide auto-scale button ('A' in lower-left corner).
        """
    
        self.buttonsHidden = True
        self.updateButtons()
        
    def showButtons(self):
        """
        Show auto-scale button ('A' in lower-left corner).
        """
        self.buttonsHidden = False
        self.updateButtons()
        
    def updateButtons(self):
        try:
            if self._exportOpts is False and self.mouseHovering and not self.buttonsHidden and not all(self.vb.autoRangeEnabled()):
                self.autoBtn.show()
            else:
                self.autoBtn.hide()
        except RuntimeError:
            pass  # this can happen if the plot has been deleted.
            
    def _plotArray(self, arr, x=None, **kargs):
        if arr.ndim != 1:
            raise ValueError(f"Array must be 1D to plot (shape is {arr.shape})")
        if x is None:
            x = np.arange(arr.shape[0])
        if x.ndim != 1:
            raise ValueError(f"X array must be 1D to plot (shape is {x.shape})")
        return PlotCurveItem(arr, x=x, **kargs)

    def setExportMode(self, export: bool, opts=None):
        """
        Set whether the item will allow export via screenshots or image files.

        If export mode is enabled, then the item will allow export and the export
        options dock will be displayed. If export mode is disabled, then no export
        options will be available and the export dock will be hidden.

        Parameters
        ----------
        export : bool
            If ``True``, the item will allow export.
        opts : dict or None, optional
            A dictionary of export options. If ``None``, the default export
            options will be used. By default, ``None``.
        """
        super().setExportMode(export, opts)
        self.updateButtons()
    
    def _chooseFilenameDialog(self, handler):
        self.fileDialog = FileDialog()
        if PlotItem.lastFileDir is not None:
            self.fileDialog.setDirectory(PlotItem.lastFileDir)
        self.fileDialog.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)
        self.fileDialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        self.fileDialog.show()
        self.fileDialog.fileSelected.connect(handler)
