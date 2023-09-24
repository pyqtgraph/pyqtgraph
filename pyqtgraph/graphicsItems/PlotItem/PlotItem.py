import collections.abc
import os
import warnings
import weakref

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
from ..ScatterPlotItem import ScatterPlotItem
from ..ViewBox import ViewBox

translate = QtCore.QCoreApplication.translate

from . import plotConfigTemplate_generic as ui_template

__all__ = ['PlotItem']


class PlotItem(GraphicsWidget):
    """GraphicsWidget implementing a standard 2D plotting area with axes.

    **Bases:** :class:`GraphicsWidget <pyqtgraph.GraphicsWidget>`
    
    This class provides the ViewBox-plus-axes that appear when using
    :func:`pg.plot() <pyqtgraph.plot>`, :class:`PlotWidget <pyqtgraph.PlotWidget>`,
    and :func:`GraphicsLayout.addPlot() <pyqtgraph.GraphicsLayout.addPlot>`.

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
    
    ==================== =======================================================================
    **Signals:**
    sigYRangeChanged     wrapped from :class:`ViewBox <pyqtgraph.ViewBox>`
    sigXRangeChanged     wrapped from :class:`ViewBox <pyqtgraph.ViewBox>`
    sigRangeChanged      wrapped from :class:`ViewBox <pyqtgraph.ViewBox>`
    ==================== =======================================================================
    """
    
    sigRangeChanged = QtCore.Signal(object, object)    ## Emitted when the ViewBox range has changed
    sigYRangeChanged = QtCore.Signal(object, object)   ## Emitted when the ViewBox Y range has changed
    sigXRangeChanged = QtCore.Signal(object, object)   ## Emitted when the ViewBox X range has changed
        
    lastFileDir = None
    
    def __init__(self, parent=None, name=None, labels=None, title=None, viewBox=None, axisItems=None, enableMenu=True, **kargs):
        """
        Create a new PlotItem. All arguments are optional.
        Any extra keyword arguments are passed to :func:`PlotItem.plot() <pyqtgraph.PlotItem.plot>`.
        
        ==============  ==========================================================================================
        **Arguments:**
        *title*         Title to display at the top of the item. Html is allowed.
        *labels*        A dictionary specifying the axis labels to display::
                   
                            {'left': (args), 'bottom': (args), ...}
                     
                        The name of each axis and the corresponding arguments are passed to 
                        :func:`PlotItem.setLabel() <pyqtgraph.PlotItem.setLabel>`
                        Optionally, PlotItem my also be initialized with the keyword arguments left,
                        right, top, or bottom to achieve the same effect.
        *name*          Registers a name for this view so that others may link to it
        *viewBox*       If specified, the PlotItem will be constructed with this as its ViewBox.
        *axisItems*     Optional dictionary instructing the PlotItem to use pre-constructed items
                        for its axes. The dict keys must be axis names ('left', 'bottom', 'right', 'top')
                        and the values must be instances of AxisItem (or at least compatible with AxisItem).
        ==============  ==========================================================================================
        """
        
        GraphicsWidget.__init__(self, parent)
        
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        
        ## Set up control buttons
        self.autoBtn = ButtonItem(icons.getGraphPixmap('auto'), 14, self)
        self.autoBtn.mode = 'auto'
        self.autoBtn.clicked.connect(self.autoBtnClicked)
        self.buttonsHidden = False ## whether the user has requested buttons to be hidden
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
        # Change these properties to adjust the appearance of the averged curve:
        self.avgPen = fn.mkPen([0, 200, 0])
        self.avgShadowPen = fn.mkPen([0, 0, 0], width=4) # the previous default of [0,0,0,100] prevent fast drawing of the wide shadow line

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
            if label in kargs:
                labels[label] = kargs[label]
                del kargs[label]
        for k in labels:
            if isinstance(labels[k], str):
                labels[k] = (labels[k],)
            self.setLabel(k, *labels[k])
                
        if title is not None:
            self.setTitle(title)
        
        if len(kargs) > 0:
            self.plot(**kargs)        
        
    def implements(self, interface=None):
        return interface in ['ViewBoxWrapper']

    def getViewBox(self):
        """Return the :class:`ViewBox <pyqtgraph.ViewBox>` contained within."""
        return self.vb
    
    ## Wrap a few methods from viewBox. 
    #Important: don't use a settattr(m, getattr(self.vb, m)) as we'd be leaving the viebox alive
    #because we had a reference to an instance method (creating wrapper methods at runtime instead).
    for m in ['setXRange', 'setYRange', 'setXLink', 'setYLink', 'setAutoPan',         # NOTE: 
              'setAutoVisible', 'setDefaultPadding', 'setRange', 'autoRange', 'viewRect', 'viewRange',     # If you update this list, please 
              'setMouseEnabled', 'setLimits', 'enableAutoRange', 'disableAutoRange',  # update the class docstring 
              'setAspectLocked', 'invertY', 'invertX', 'register', 'unregister']:                # as well.
                
        def _create_method(name):
            def method(self, *args, **kwargs):
                return getattr(self.vb, name)(*args, **kwargs)
            method.__name__ = name
            return method
        
        locals()[m] = _create_method(m)
        
    del _create_method
    
    def setAxisItems(self, axisItems=None):
        """
        Place axis items as given by `axisItems`. Initializes non-existing axis items.
        
        ==============  ==========================================================================================
        **Arguments:**
        *axisItems*     Optional dictionary instructing the PlotItem to use pre-constructed items
                        for its axes. The dict keys must be axis names ('left', 'bottom', 'right', 'top')
                        and the values must be instances of AxisItem (or at least compatible with AxisItem).
        ==============  ==========================================================================================
        """
        
        if axisItems is None:
            axisItems = {}
        
        # Array containing visible axis items
        # Also containing potentially hidden axes, but they are not touched so it does not matter
        visibleAxes = ['left', 'bottom']
        visibleAxes.extend(axisItems.keys()) # Note that it does not matter that this adds
                                             # some values to visibleAxes a second time
        
        for k, pos in (('top', (1,1)), ('bottom', (3,1)), ('left', (2,0)), ('right', (2,2))):
            if k in self.axes:
                if k not in axisItems:
                    continue # Nothing to do here
                
                # Remove old axis
                oldAxis = self.axes[k]['item']
                self.layout.removeItem(oldAxis)
                oldAxis.scene().removeItem(oldAxis)
                oldAxis.unlinkFromView()
            
            # Create new axis
            if k in axisItems:
                axis = axisItems[k]
                if axis.scene() is not None:
                    if k not in self.axes or axis != self.axes[k]["item"]:
                        raise RuntimeError(
                            "Can't add an axis to multiple plots. Shared axes"
                            " can be achieved with multiple AxisItem instances"
                            " and set[X/Y]Link.")
            else:
                axis = AxisItem(orientation=k, parent=self)
            
            # Set up new axis
            axis.linkToView(self.vb)
            self.axes[k] = {'item': axis, 'pos': pos}
            self.layout.addItem(axis, *pos)
            # place axis above images at z=0, items that want to draw over the axes should be placed at z>=1:
            axis.setZValue(0.5) 
            axis.setFlag(axis.GraphicsItemFlag.ItemNegativeZStacksBehindParent)           
            axisVisible = k in visibleAxes
            self.showAxis(k, axisVisible)
        
    def setLogMode(self, x=None, y=None):
        """
        Set log scaling for `x` and/or `y` axes.
        This informs PlotDataItems to transform logarithmically and switches
        the axes to use log ticking. 
        
        Note that *no other items* in the scene will be affected by
        this; there is (currently) no generic way to redisplay a GraphicsItem
        with log coordinates.
        
        """
        if x is not None:
            self.ctrl.logXCheck.setChecked(x)
        if y is not None:
            self.ctrl.logYCheck.setChecked(y)
        
    def showGrid(self, x=None, y=None, alpha=None):
        """
        Show or hide the grid for either axis.
        
        ==============  =====================================
        **Arguments:**
        x               (bool) Whether to show the X grid
        y               (bool) Whether to show the Y grid
        alpha           (0.0-1.0) Opacity of the grid
        ==============  =====================================
        """
        if x is None and y is None and alpha is None:
            raise Exception("Must specify at least one of x, y, or alpha.")  ## prevent people getting confused if they just call showGrid()
        
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
        
    def updateGrid(self, *args):
        alpha = self.ctrl.gridAlphaSlider.value()
        x = alpha if self.ctrl.xGridCheck.isChecked() else False
        y = alpha if self.ctrl.yGridCheck.isChecked() else False
        self.getAxis('top').setGrid(x)
        self.getAxis('bottom').setGrid(x)
        self.getAxis('left').setGrid(y)
        self.getAxis('right').setGrid(y)

    def viewGeometry(self):
        """Return the screen geometry of the viewbox"""
        v = self.scene().views()[0]
        b = self.vb.mapRectToScene(self.vb.boundingRect())
        wr = v.mapFromScene(b).boundingRect()
        pos = v.mapToGlobal(v.pos())
        wr.adjust(pos.x(), pos.y(), pos.x(), pos.y())
        return wr

    def avgToggled(self, b):
        if b:
            self.recomputeAverages()
        for k in self.avgCurves:
            self.avgCurves[k][1].setVisible(b)
        
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
        
            ### First determine the key of the curve to which this new data should be averaged
            for i in range(self.ctrl.avgParamList.count()):
                item = self.ctrl.avgParamList.item(i)
                if item.checkState() == QtCore.Qt.CheckState.Checked:
                    remKeys.append(str(item.text()))
                else:
                    addKeys.append(str(item.text()))
                    
            if len(remKeys) < 1:  ## In this case, there would be 1 average plot for each data plot; not useful.
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
        
    def autoBtnClicked(self):
        if self.autoBtn.mode == 'auto':
            self.enableAutoRange()
            self.autoBtn.hide()
        else:
            self.disableAutoRange()
            
    def viewStateChanged(self):
        self.updateButtons()

    def addItem(self, item, *args, **kargs):
        """
        Add a graphics item to the view box. 
        If the item has plot data (:class:`PlotDataItem <pyqtgraph.PlotDataItem>` , 
        :class:`~pyqtgraph.PlotCurveItem` , :class:`~pyqtgraph.ScatterPlotItem` ), 
        it may be included in analysis performed by the PlotItem.
        """
        if item in self.items:
            warnings.warn('Item already added to PlotItem, ignoring.')
            return
        self.items.append(item)
        vbargs = {}
        if 'ignoreBounds' in kargs:
            vbargs['ignoreBounds'] = kargs['ignoreBounds']
        self.vb.addItem(item, *args, **vbargs)
        name = None
        if hasattr(item, 'implements') and item.implements('plotData'):
            name = item.name()
            self.dataItems.append(item)
            #self.plotChanged()
            
            params = kargs.get('params', {})
            self.itemMeta[item] = params
            #item.setMeta(params)
            self.curves.append(item)
            #self.addItem(c)
            
        if hasattr(item, 'setLogMode'):
            item.setLogMode(self.ctrl.logXCheck.isChecked(), self.ctrl.logYCheck.isChecked())
            
        if isinstance(item, PlotDataItem):
            ## configure curve for this plot
            (alpha, auto) = self.alphaState()
            item.setAlpha(alpha, auto)
            item.setFftMode(self.ctrl.fftCheck.isChecked())
            item.setDownsampling(*self.downsampleMode())
            item.setClipToView(self.clipToViewMode())
            
            ## Hide older plots if needed
            self.updateDecimation()
            
            ## Add to average if needed
            self.updateParamList()
            if self.ctrl.averageGroup.isChecked() and 'skipAverage' not in kargs:
                self.addAvgCurve(item)
                
            #c.connect(c, QtCore.SIGNAL('plotChanged'), self.plotChanged)
            #item.sigPlotChanged.connect(self.plotChanged)
            #self.plotChanged()
        #name = kargs.get('name', getattr(item, 'opts', {}).get('name', None))
        if name is not None and hasattr(self, 'legend') and self.legend is not None:
            self.legend.addItem(item, name=name)            

    def listDataItems(self):
        """Return a list of all data items (:class:`PlotDataItem <pyqtgraph.PlotDataItem>`, 
        :class:`~pyqtgraph.PlotCurveItem` , :class:`~pyqtgraph.ScatterPlotItem` , etc)
        contained in this PlotItem."""
        return self.dataItems[:]

    def addLine(self, x=None, y=None, z=None, **kwds):
        """
        Create an :class:`~pyqtgraph.InfiniteLine` and add to the plot. 
        
        If `x` is specified,
        the line will be vertical. If `y` is specified, the line will be
        horizontal. All extra keyword arguments are passed to
        :func:`InfiniteLine.__init__() <pyqtgraph.InfiniteLine.__init__>`.
        Returns the item created.
        """
        kwds['pos'] = kwds.get('pos', x if x is not None else y)
        kwds['angle'] = kwds.get('angle', 0 if x is None else 90)
        line = InfiniteLine(**kwds)
        self.addItem(line)
        if z is not None:
            line.setZValue(z)
        return line

    def removeItem(self, item):
        """
        Remove an item from the PlotItem's :class:`~pyqtgraph.ViewBox`.
        """
        if not item in self.items:
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
        for i in self.curves[:]:
            self.removeItem(i)
        self.avgCurves = {}        
    
    def plot(self, *args, **kargs):
#        **Additional arguments:**
        """
        Add and return a new plot.
        See :func:`PlotDataItem.__init__ <pyqtgraph.PlotDataItem.__init__>` for data arguments
        
        **Additional allowed arguments**
        
        ========= =================================================================
        `clear`   clears all plots before displaying new data
        `params`  sets meta-parameters to associate with this data
        ========= =================================================================
        """
        clear = kargs.get('clear', False)
        params = kargs.get('params', None)
          
        if clear:
            self.clear()
            
        item = PlotDataItem(*args, **kargs)
            
        if params is None:
            params = {}
        self.addItem(item, params=params)
        
        return item

    def addLegend(self, offset=(30, 30), **kwargs):
        """
        Create a new :class:`~pyqtgraph.LegendItem` and anchor it over the internal 
        :class:`~pyqtgraph.ViewBox`. Plots added after this will be automatically 
        displayed in the legend if they are created with a 'name' argument.

        If a :class:`~pyqtgraph.LegendItem` has already been created using this method, 
        that item will be returned rather than creating a new one.

        Accepts the same arguments as :func:`~pyqtgraph.LegendItem.__init__`.
        """

        if self.legend is None:
            self.legend = LegendItem(offset=offset, **kwargs)
            self.legend.setParentItem(self.vb)
        return self.legend
        
    def addColorBar(self, image, **kargs):
        """
        Adds a color bar linked to the ImageItem specified by `image`.
        AAdditional parameters will be passed to the `pyqtgraph.ColorBarItem`.
        
        A call like `plot.addColorBar(img, colorMap='viridis')` is a convenient
        method to assign and show a color map.
        """
        from ..ColorBarItem import ColorBarItem  # avoid circular import
        bar = ColorBarItem(**kargs)
        bar.setImageItem( image, insert_in=self )
        return bar

    def multiDataPlot(self, *, x=None, y=None, constKwargs=None, **kwargs):
        """
        Allow plotting multiple curves on the same plot, changing some kwargs
        per curve.

        Parameters
        ----------
        x, y: array_like
            can be in the following formats:
              - {x or y} = [n1, n2, n3, ...]: The named argument iterates through
                ``n`` curves, while the unspecified argument is range(len(n)) for
                each curve.
              - x, [y1, y2, y3, ...]
              - [x1, x2, x3, ...], [y1, y2, y3, ...]
              - [x1, x2, x3, ...], y

              where ``x_n`` and ``y_n`` are ``ndarray`` data for each curve. Since
              ``x`` and ``y`` values are matched using ``zip``, unequal lengths mean
              the longer array will be truncated. Note that 2D matrices for either x
              or y are considered lists of curve
              data.
        constKwargs: dict, optional
            A dict of {str: value} passed to each curve during ``plot()``.
        kwargs: dict, optional
            A dict of {str: iterable} where the str is the name of a kwarg and the
            iterable is a list of values, one for each plotted curve.
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
        xy: 'dict[str, list | None]' = dict(x=x, y=y)
        for key, oppositeVal in zip(('x', 'y'), [y, x]):
            oppositeVal: 'Iterable | None'
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
            kwargsi = {kk: kwargs[kk][ii] for kk in kwargs}
            constKwargs.update(kwargsi)
            curves.append(self.plot(xi, yi, **constKwargs))
        return curves

    def scatterPlot(self, *args, **kargs):
        if 'pen' in kargs:
            kargs['symbolPen'] = kargs['pen']
        kargs['pen'] = None
            
        if 'brush' in kargs:
            kargs['symbolBrush'] = kargs['brush']
            del kargs['brush']
            
        if 'size' in kargs:
            kargs['symbolSize'] = kargs['size']
            del kargs['size']

        return self.plot(*args, **kargs)
                
    def replot(self):
        self.update()

    def updateParamList(self):
        self.ctrl.avgParamList.clear()
        ## Check to see that each parameter for each curve is present in the list
        for c in self.curves:
            for p in list(self.itemMeta.get(c, {}).keys()):
                if type(p) is tuple:
                    p = '.'.join(p)
                    
                ## If the parameter is not in the list, add it.
                matches = self.ctrl.avgParamList.findItems(p, QtCore.Qt.MatchFlag.MatchExactly)
                if len(matches) == 0:
                    i = QtWidgets.QListWidgetItem(p)
                    if p in self.paramList and self.paramList[p] is True:
                        i.setCheckState(QtCore.Qt.CheckState.Checked)
                    else:
                        i.setCheckState(QtCore.Qt.CheckState.Unchecked)
                    self.ctrl.avgParamList.addItem(i)
                else:
                    i = matches[0]
                    
                self.paramList[p] = (i.checkState() == QtCore.Qt.CheckState.Checked)

    def writeSvg(self, fileName=None):
        if fileName is None:
            self._chooseFilenameDialog(handler=self.writeSvg)
            return

        fileName = str(fileName)
        PlotItem.lastFileDir = os.path.dirname(fileName)
        
        from ...exporters import SVGExporter
        ex = SVGExporter(self)
        ex.export(fileName)
        
    def writeImage(self, fileName=None):
        if fileName is None:
            self._chooseFilenameDialog(handler=self.writeImage)
            return

        from ...exporters import ImageExporter
        ex = ImageExporter(self)
        ex.export(fileName)
        
    def writeCsv(self, fileName=None):
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
      
    def updateSpectrumMode(self, b=None):
        if b is None:
            b = self.ctrl.fftCheck.isChecked()
        for c in self.curves:
            c.setFftMode(b)
        self.enableAutoRange()
        self.recomputeAverages()
            
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
    
    def updateDerivativeMode(self):
        d = self.ctrl.derivativeCheck.isChecked()
        for i in self.items:
            if hasattr(i, 'setDerivativeMode'):
                i.setDerivativeMode(d)
        self.enableAutoRange()
        self.recomputeAverages()

    def updatePhasemapMode(self):
        d = self.ctrl.phasemapCheck.isChecked()
        for i in self.items:
            if hasattr(i, 'setPhasemapMode'):
                i.setPhasemapMode(d)
        self.enableAutoRange()
        self.recomputeAverages()
        
        
    def setDownsampling(self, ds=None, auto=None, mode=None):
        """
        Changes the default downsampling mode for all :class:`~pyqtgraph.PlotDataItem` managed by this plot.
        
        =============== ====================================================================
        **Arguments:**
        ds              (int) Reduce visible plot samples by this factor, or

                        (bool) To enable/disable downsampling without changing the value.

        auto            (bool) If `True`, automatically pick ``ds`` based on visible range

        mode            'subsample': Downsample by taking the first of N samples. This 
                        method is fastest but least accurate.

                        'mean': Downsample by taking the mean of N samples.

                        'peak': Downsample by drawing a saw wave that follows the min and 
                        max of the original data. This method produces the best visual 
                        representation of the data but is slower.
        =============== ====================================================================
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
                raise ValueError("mode argument must be 'subsample', 'mean', or 'peak'.")
            
    def updateDownsampling(self):
        ds, auto, method = self.downsampleMode()
        clip = self.ctrl.clipToViewCheck.isChecked()
        for c in self.curves:
            c.setDownsampling(ds, auto, method)
            c.setClipToView(clip)
        self.recomputeAverages()
        
    def downsampleMode(self):
        if self.ctrl.downsampleCheck.isChecked():
            ds = self.ctrl.downsampleSpin.value()
        else:
            ds = 1
            
        auto = self.ctrl.downsampleCheck.isChecked() and self.ctrl.autoDownsampleCheck.isChecked()
            
        if self.ctrl.subsampleRadio.isChecked():
            method = 'subsample' 
        elif self.ctrl.meanRadio.isChecked():
            method = 'mean'
        elif self.ctrl.peakRadio.isChecked():
            method = 'peak'
        else:
            raise ValueError("one of the method radios must be selected for: 'subsample', 'mean', or 'peak'.")
        
        return ds, auto, method
        
    def setClipToView(self, clip):
        """Set the default clip-to-view mode for all :class:`~pyqtgraph.PlotDataItem` s managed by this plot.
        If *clip* is `True`, then PlotDataItems will attempt to draw only points within the visible
        range of the ViewBox."""
        self.ctrl.clipToViewCheck.setChecked(clip)
        
    def clipToViewMode(self):
        return self.ctrl.clipToViewCheck.isChecked()
    
    def _handle_max_traces_toggle(self, check_state):
        if check_state:
            self.updateDecimation()
        else:
            for curve in self.curves:
                curve.show()
    
    def updateDecimation(self):
        """
        Reduce or increase number of visible curves according to value set by the `Max Traces` spinner,
        if `Max Traces` is checked in the context menu. Destroy curves that are not visible if 
        `forget traces` is checked. In most cases, this function is called automaticaly when the 
        `Max Traces` GUI elements are triggered. It is also alled when the state of PlotItem is updated,
        its state is restored, or new items added added/removed.
        
        This can cause an unexpected or conflicting state of curve visibility (or destruction) if curve
        visibilities are controlled externally. In the case of external control it is advised to disable
        the `Max Traces` checkbox (or context menu) to prevent unexpected curve state changes.
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
            if self.ctrl.autoPointsCheck.isChecked():
                mode = None
            else:
                mode = True
        else:
            mode = False
        return mode

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
        By default, the ViewBox's context menu will also be affected.
        (use ``enableViewBoxMenu=None`` to leave the ViewBox unchanged)
        """
        self._menuEnabled = enableMenu
        if enableViewBoxMenu is None:
            return
        if enableViewBoxMenu == 'same':
            enableViewBoxMenu = enableMenu
        self.vb.setMenuEnabled(enableViewBoxMenu)
    
    def menuEnabled(self):
        return self._menuEnabled

    def setContextMenuActionVisible(self, name : str, visible : bool) -> None:
        """
        Changes the context menu action visibility

        Parameters
        ----------
        name: str
            Action name
            must be one of 'Transforms', 'Downsample', 'Average','Alpha', 'Grid', or 'Points'
        visible: bool
            Determines if action will be display
            True action is visible
            False action is invisible.
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
            raise Exception("Scale '%s' not found. Scales are: %s" % (key, str(list(self.axes.keys()))))
        
    def getScale(self, key):
        return self.getAxis(key)
        
    def getAxis(self, name):
        """Return the specified AxisItem. 
        *name* should be 'left', 'bottom', 'top', or 'right'."""
        self._checkScaleKey(name)
        return self.axes[name]['item']
        
    def setLabel(self, axis, text=None, units=None, unitPrefix=None, **args):
        """
        Sets the label for an axis. Basic HTML formatting is allowed.
        
        ==============  =================================================================
        **Arguments:**
        axis            must be one of 'left', 'bottom', 'right', or 'top'
        text            text to display along the axis. HTML allowed.
        units           units to display after the title. If units are given,
                        then an SI prefix will be automatically appended
                        and the axis values will be scaled accordingly.
                        (ie, use 'V' instead of 'mV'; 'm' will be added automatically)
        ==============  =================================================================
        """
        self.getAxis(axis).setLabel(text=text, units=units, **args)
        self.showAxis(axis)
        
    def setLabels(self, **kwds):
        """
        Convenience function allowing multiple labels and/or title to be set in one call.
        Keyword arguments can be 'title', 'left', 'bottom', 'right', or 'top'.
        Values may be strings or a tuple of arguments to pass to :func:`setLabel`.
        """
        for k,v in kwds.items():
            if k == 'title':
                self.setTitle(v)
            else:
                if isinstance(v, str):
                    v = (v,)
                self.setLabel(k, *v)
        
    def showLabel(self, axis, show=True):
        """
        Show or hide one of the plot's axis labels (the axis itself will be unaffected).
        axis must be one of 'left', 'bottom', 'right', or 'top'
        """
        self.getScale(axis).showLabel(show)

    def setTitle(self, title=None, **args):
        """
        Set the title of the plot. Basic HTML formatting is allowed.
        If title is None, then the title will be hidden.
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

    def showAxis(self, axis, show=True):
        """
        Show or hide one of the plot's axes.
        axis must be one of 'left', 'bottom', 'right', or 'top'
        """
        s = self.getScale(axis)
        if show:
            s.show()
        else:
            s.hide()
            
    def hideAxis(self, axis):
        """Hide one of the PlotItem's axes. ('left', 'bottom', 'right', or 'top')"""
        self.showAxis(axis, False)
        
    def showAxes(self, selection, showValues=True, size=False):
        """ 
        Convenience method for quickly configuring axis settings.
        
        Parameters
        ----------
        selection: bool or tuple of bool 
            Determines which AxisItems will be displayed.
            If in tuple form, order is (left, top, right, bottom)
            A single boolean value will set all axes, 
            so that ``showAxes(True)`` configures the axes to draw a frame.
        showValues: bool or tuple of bool, optional
            Determines if values will be displayed for the ticks of each axis.
            True value shows values for left and bottom axis (default).
            False shows no values.
            If in tuple form, order is (left, top, right, bottom)
            None leaves settings unchanged.
            If not specified, left and bottom axes will be drawn with values.
        size: float or tuple of float, optional
            Reserves as fixed amount of space (width for vertical axis, height for horizontal axis)
            for each axis where tick values are enabled. If only a single float value is given, it
            will be applied for both width and height. If `None` is given instead of a float value,
            the axis reverts to automatic allocation of space.
            If in tuple form, order is (width, height)
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
            if show_axis is None:
                pass # leave axis display as it is.
            else:
                if show_axis: self.showAxis(axis_key)
                else        : self.hideAxis(axis_key)
                
            if show_value is None:
                pass # leave value display as it is.
            else:
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
        """Causes auto-scale button ('A' in lower-left corner) to be hidden for this PlotItem"""
        #self.ctrlBtn.hide()
        self.buttonsHidden = True
        self.updateButtons()
        
    def showButtons(self):
        """Causes auto-scale button ('A' in lower-left corner) to be visible for this PlotItem"""
        #self.ctrlBtn.hide()
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
            raise Exception("Array must be 1D to plot (shape is %s)" % arr.shape)
        if x is None:
            x = np.arange(arr.shape[0])
        if x.ndim != 1:
            raise Exception("X array must be 1D to plot (shape is %s)" % x.shape)
        c = PlotCurveItem(arr, x=x, **kargs)
        return c
        
    def _plotMetaArray(self, arr, x=None, autoLabel=True, **kargs):
        if arr.ndim != 1:
            raise Exception('can only automatically plot 1 dimensional arrays.')
        ## create curve
        try:
            xv = arr.xvals(0)
        except:
            if x is None:
                xv = np.arange(arr.shape[0])
            else:
                xv = x
        c = PlotCurveItem(**kargs)
        c.setData(x=xv, y=arr.view(np.ndarray))
        
        if autoLabel:
            name = arr._info[0].get('name', None)
            units = arr._info[0].get('units', None)
            self.setLabel('bottom', text=name, units=units)
            
            name = arr._info[1].get('name', None)
            units = arr._info[1].get('units', None)
            self.setLabel('left', text=name, units=units)
            
        return c
      
    def setExportMode(self, export, opts=None):
        GraphicsWidget.setExportMode(self, export, opts)
        self.updateButtons()
    
    def _chooseFilenameDialog(self, handler):
        self.fileDialog = FileDialog()
        if PlotItem.lastFileDir is not None:
            self.fileDialog.setDirectory(PlotItem.lastFileDir)
        self.fileDialog.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)
        self.fileDialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        self.fileDialog.show()
        self.fileDialog.fileSelected.connect(handler)
