# -*- coding: utf-8 -*-
"""
PlotItem.py -  Graphics item implementing a scalable ViewBox with plotting powers.
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more infomation.

This class is one of the workhorses of pyqtgraph. It implements a graphics item with 
plots, labels, and scales which can be viewed inside a QGraphicsScene. If you want
a widget that can be added to your GUI, see PlotWidget instead.

This class is very heavily featured:
  - Automatically creates and manages PlotCurveItems
  - Fast display and update of plots
  - Manages zoom/pan ViewBox, scale, and label elements
  - Automatic scaling when data changes
  - Control panel with a huge feature set including averaging, decimation,
    display, power spectrum, svg/png export, plot linking, and more.
"""
#from graphicsItems import *
from .plotConfigTemplate import *
from pyqtgraph.Qt import QtGui, QtCore, QtSvg
import pyqtgraph.functions as fn
from pyqtgraph.widgets.FileDialog import FileDialog
import weakref
#from types import *
import numpy as np
import os
#from .. PlotCurveItem import PlotCurveItem
#from .. ScatterPlotItem import ScatterPlotItem
from .. PlotDataItem import PlotDataItem
from .. ViewBox import ViewBox
from .. AxisItem import AxisItem
from .. LabelItem import LabelItem
from .. GraphicsWidget import GraphicsWidget
from .. ButtonItem import ButtonItem
from pyqtgraph.WidgetGroup import WidgetGroup
import collections

__all__ = ['PlotItem']

#try:
    #from WidgetGroup import *
    #HAVE_WIDGETGROUP = True
#except:
    #HAVE_WIDGETGROUP = False
    
try:
    from metaarray import *
    HAVE_METAARRAY = True
except:
    HAVE_METAARRAY = False




class PlotItem(GraphicsWidget):
    
    """
    **Bases:** :class:`GraphicsWidget <pyqtgraph.GraphicsWidget>`
    
    Plot graphics item that can be added to any graphics scene. Implements axes, titles, and interactive viewbox. 
    PlotItem also provides some basic analysis functionality that may be accessed from the context menu.
    Use :func:`plot() <pyqtgraph.PlotItem.plot>` to create a new PlotDataItem and add it to the view.
    Use :func:`addItem() <pyqtgraph.PlotItem.addItem>` to add any QGraphicsItem to the view.
    
    This class wraps several methods from its internal ViewBox:
    :func:`setXRange <pyqtgraph.ViewBox.setXRange>`,
    :func:`setYRange <pyqtgraph.ViewBox.setYRange>`,
    :func:`setRange <pyqtgraph.ViewBox.setRange>`,
    :func:`autoRange <pyqtgraph.ViewBox.autoRange>`,
    :func:`setXLink <pyqtgraph.ViewBox.setXLink>`,
    :func:`setYLink <pyqtgraph.ViewBox.setYLink>`,
    :func:`setAutoPan <pyqtgraph.ViewBox.setAutoPan>`,
    :func:`setAutoVisible <pyqtgraph.ViewBox.setAutoVisible>`,
    :func:`viewRect <pyqtgraph.ViewBox.viewRect>`,
    :func:`viewRange <pyqtgraph.ViewBox.viewRange>`,
    :func:`setMouseEnabled <pyqtgraph.ViewBox.setMouseEnabled>`,
    :func:`enableAutoRange <pyqtgraph.ViewBox.enableAutoRange>`,
    :func:`disableAutoRange <pyqtgraph.ViewBox.disableAutoRange>`,
    :func:`setAspectLocked <pyqtgraph.ViewBox.setAspectLocked>`,
    :func:`register <pyqtgraph.ViewBox.register>`,
    :func:`unregister <pyqtgraph.ViewBox.unregister>`
    
    The ViewBox itself can be accessed by calling :func:`getViewBox() <pyqtgraph.PlotItem.getViewBox>` 
    
    ==================== =======================================================================
    **Signals**
    sigYRangeChanged     wrapped from :class:`ViewBox <pyqtgraph.ViewBox>`
    sigXRangeChanged     wrapped from :class:`ViewBox <pyqtgraph.ViewBox>`
    sigRangeChanged      wrapped from :class:`ViewBox <pyqtgraph.ViewBox>`
    ==================== =======================================================================
    """
    
    sigRangeChanged = QtCore.Signal(object, object)    ## Emitted when the ViewBox range has changed
    sigYRangeChanged = QtCore.Signal(object, object)   ## Emitted when the ViewBox Y range has changed
    sigXRangeChanged = QtCore.Signal(object, object)   ## Emitted when the ViewBox X range has changed
    
    
    lastFileDir = None
    managers = {}
    
    def __init__(self, parent=None, name=None, labels=None, title=None, **kargs):
        """
        Create a new PlotItem. All arguments are optional.
        Any extra keyword arguments are passed to PlotItem.plot().
        
        =============  ==========================================================================================
        **Arguments**
        *title*        Title to display at the top of the item. Html is allowed.
        *labels*       A dictionary specifying the axis labels to display::
                   
                           {'left': (args), 'bottom': (args), ...}
                     
                       The name of each axis and the corresponding arguments are passed to 
                       :func:`PlotItem.setLabel() <pyqtgraph.PlotItem.setLabel>`
                       Optionally, PlotItem my also be initialized with the keyword arguments left,
                       right, top, or bottom to achieve the same effect.
        *name*         Registers a name for this view so that others may link to it  
        =============  ==========================================================================================
            
            
        """
        
        GraphicsWidget.__init__(self, parent)
        
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        
        ## Set up control buttons
        path = os.path.dirname(__file__)
        #self.ctrlBtn = ButtonItem(os.path.join(path, 'ctrl.png'), 14, self)
        #self.ctrlBtn.clicked.connect(self.ctrlBtnClicked)
        self.autoImageFile = os.path.join(path, 'auto.png')
        self.lockImageFile = os.path.join(path, 'lock.png')
        self.autoBtn = ButtonItem(self.autoImageFile, 14, self)
        self.autoBtn.mode = 'auto'
        self.autoBtn.clicked.connect(self.autoBtnClicked)
        
        self.layout = QtGui.QGraphicsGridLayout()
        self.layout.setContentsMargins(1,1,1,1)
        self.setLayout(self.layout)
        self.layout.setHorizontalSpacing(0)
        self.layout.setVerticalSpacing(0)
        
        self.vb = ViewBox(name=name)
        self.vb.sigRangeChanged.connect(self.sigRangeChanged)
        self.vb.sigXRangeChanged.connect(self.sigXRangeChanged)
        self.vb.sigYRangeChanged.connect(self.sigYRangeChanged)
        #self.vb.sigRangeChangedManually.connect(self.enableManualScale)
        #self.vb.sigRangeChanged.connect(self.viewRangeChanged)
        
        self.layout.addItem(self.vb, 2, 1)
        self.alpha = 1.0
        self.autoAlpha = True
        self.spectrumMode = False
        
        #self.autoScale = [True, True]
        
        ## Create and place scale items
        self.scales = {
            'top':    {'item': AxisItem(orientation='top',    linkView=self.vb), 'pos': (1, 1)}, 
            'bottom': {'item': AxisItem(orientation='bottom', linkView=self.vb), 'pos': (3, 1)}, 
            'left':   {'item': AxisItem(orientation='left',   linkView=self.vb), 'pos': (2, 0)}, 
            'right':  {'item': AxisItem(orientation='right',  linkView=self.vb), 'pos': (2, 2)}
        }
        for k in self.scales:
            item = self.scales[k]['item']
            self.layout.addItem(item, *self.scales[k]['pos'])
            item.setZValue(-1000)
            item.setFlag(item.ItemNegativeZStacksBehindParent)
        
        self.titleLabel = LabelItem('', size='11pt')
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
        

        ## Wrap a few methods from viewBox
        for m in [
            'setXRange', 'setYRange', 'setXLink', 'setYLink', 'setAutoPan', 'setAutoVisible',
            'setRange', 'autoRange', 'viewRect', 'viewRange', 'setMouseEnabled',
            'enableAutoRange', 'disableAutoRange', 'setAspectLocked',
            'register', 'unregister']:  ## NOTE: If you update this list, please update the class docstring as well.
            setattr(self, m, getattr(self.vb, m))
            
        self.items = []
        self.curves = []
        self.itemMeta = weakref.WeakKeyDictionary()
        self.dataItems = []
        self.paramList = {}
        self.avgCurves = {}
        
        ### Set up context menu
        
        w = QtGui.QWidget()
        self.ctrl = c = Ui_Form()
        c.setupUi(w)
        dv = QtGui.QDoubleValidator(self)
        
        menuItems = [
            ('Transforms', c.transformGroup),
            ('Downsample', c.decimateGroup),
            ('Average', c.averageGroup),
            ('Alpha', c.alphaGroup),
            ('Grid', c.gridGroup),
            ('Points', c.pointsGroup),
        ]
        
        
        self.ctrlMenu = QtGui.QMenu()
        
        self.ctrlMenu.setTitle('Plot Options')
        self.subMenus = []
        for name, grp in menuItems:
            sm = QtGui.QMenu(name)
            act = QtGui.QWidgetAction(self)
            act.setDefaultWidget(grp)
            sm.addAction(act)
            self.subMenus.append(sm)
            self.ctrlMenu.addMenu(sm)
        
        ## exporting is handled by GraphicsScene now
        #exportOpts = collections.OrderedDict([
            #('SVG - Full Plot', self.saveSvgClicked),
            #('SVG - Curves Only', self.saveSvgCurvesClicked),
            #('Image', self.saveImgClicked),
            #('CSV', self.saveCsvClicked),
        #])
        
        #self.vb.menu.setExportMethods(exportOpts)
        
        
        #if HAVE_WIDGETGROUP:
        self.stateGroup = WidgetGroup()
        for name, w in menuItems:
            self.stateGroup.autoAdd(w)
        
        self.fileDialog = None
        
        #self.xLinkPlot = None
        #self.yLinkPlot = None
        #self.linksBlocked = False

        #self.setAcceptHoverEvents(True)
        
        ## Connect control widgets
        #c.xMinText.editingFinished.connect(self.setManualXScale)
        #c.xMaxText.editingFinished.connect(self.setManualXScale)
        #c.yMinText.editingFinished.connect(self.setManualYScale)
        #c.yMaxText.editingFinished.connect(self.setManualYScale)
        
        #c.xManualRadio.clicked.connect(lambda: self.updateXScale())
        #c.yManualRadio.clicked.connect(lambda: self.updateYScale())
        
        #c.xAutoRadio.clicked.connect(self.updateXScale)
        #c.yAutoRadio.clicked.connect(self.updateYScale)

        #c.xAutoPercentSpin.valueChanged.connect(self.replot)
        #c.yAutoPercentSpin.valueChanged.connect(self.replot)
        
        c.alphaGroup.toggled.connect(self.updateAlpha)
        c.alphaSlider.valueChanged.connect(self.updateAlpha)
        c.autoAlphaCheck.toggled.connect(self.updateAlpha)

        c.xGridCheck.toggled.connect(self.updateGrid)
        c.yGridCheck.toggled.connect(self.updateGrid)
        c.gridAlphaSlider.valueChanged.connect(self.updateGrid)

        c.fftCheck.toggled.connect(self.updateSpectrumMode)
        c.logXCheck.toggled.connect(self.updateLogMode)
        c.logYCheck.toggled.connect(self.updateLogMode)
        #c.saveSvgBtn.clicked.connect(self.saveSvgClicked)
        #c.saveSvgCurvesBtn.clicked.connect(self.saveSvgCurvesClicked)
        #c.saveImgBtn.clicked.connect(self.saveImgClicked)
        #c.saveCsvBtn.clicked.connect(self.saveCsvClicked)
        
        #self.ctrl.xLinkCombo.currentIndexChanged.connect(self.xLinkComboChanged)
        #self.ctrl.yLinkCombo.currentIndexChanged.connect(self.yLinkComboChanged)

        c.downsampleSpin.valueChanged.connect(self.updateDownsampling)

        self.ctrl.avgParamList.itemClicked.connect(self.avgParamListClicked)
        self.ctrl.averageGroup.toggled.connect(self.avgToggled)
        
        self.ctrl.maxTracesCheck.toggled.connect(self.updateDecimation)
        self.ctrl.maxTracesSpin.valueChanged.connect(self.updateDecimation)
        #c.xMouseCheck.toggled.connect(self.mouseCheckChanged)
        #c.yMouseCheck.toggled.connect(self.mouseCheckChanged)

        #self.xLinkPlot = None
        #self.yLinkPlot = None
        #self.linksBlocked = False
        self.manager = None
        
        self.hideAxis('right')
        self.hideAxis('top')
        self.showAxis('left')
        self.showAxis('bottom')
        
        #if name is not None:
            #self.registerPlot(name)
        if labels is None:
            labels = {}
        for label in list(self.scales.keys()):
            if label in kargs:
                labels[label] = kargs[label]
                del kargs[label]
        for k in labels:
            if isinstance(labels[k], basestring):
                labels[k] = (labels[k],)
            self.setLabel(k, *labels[k])
                
        if title is not None:
            self.setTitle(title)
        
        if len(kargs) > 0:
            self.plot(**kargs)
        
        #self.enableAutoRange()
        
    def implements(self, interface=None):
        return interface in ['ViewBoxWrapper']

    def getViewBox(self):
        """Return the ViewBox within."""
        return self.vb
    
    def setLogMode(self, x, y):
        """
        Set log scaling for x and y axes.
        This informs PlotDataItems to transform logarithmically and switches
        the axes to use log ticking. 
        
        Note that *no other items* in the scene will be affected by
        this; there is no generic way to redisplay a GraphicsItem
        with log coordinates.
        
        """
        self.ctrl.logXCheck.setChecked(x)
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
            v = np.clip(alpha, 0, 1)*self.ctrl.gridAlphaSlider.maximum()
            self.ctrl.gridAlphaSlider.setValue(v)
        
    #def paint(self, *args):
        #prof = debug.Profiler('PlotItem.paint', disabled=True)
        #QtGui.QGraphicsWidget.paint(self, *args)
        #prof.finish()
        
    ## bad idea. 
    #def __getattr__(self, attr):  ## wrap ms
        #return getattr(self.vb, attr)
        
    def close(self):
        #print "delete", self
        ## Most of this crap is needed to avoid PySide trouble. 
        ## The problem seems to be whenever scene.clear() leads to deletion of widgets (either through proxies or qgraphicswidgets)
        ## the solution is to manually remove all widgets before scene.clear() is called
        if self.ctrlMenu is None: ## already shut down
            return
        self.ctrlMenu.setParent(None)
        self.ctrlMenu = None
        
        #self.ctrlBtn.setParent(None)
        #self.ctrlBtn = None
        #self.autoBtn.setParent(None)
        #self.autoBtn = None
        
        for k in self.scales:
            i = self.scales[k]['item']
            i.close()
            
        self.scales = None
        self.scene().removeItem(self.vb)
        self.vb = None
        
        ## causes invalid index errors:
        #for i in range(self.layout.count()):
            #self.layout.removeAt(i)
            
        #for p in self.proxies:
            #try:
                #p.setWidget(None)
            #except RuntimeError:
                #break
            #self.scene().removeItem(p)
        #self.proxies = []
        
        #self.menuAction.releaseWidget(self.menuAction.defaultWidget())
        #self.menuAction.setParent(None)
        #self.menuAction = None
        
        #if self.manager is not None:
            #self.manager.sigWidgetListChanged.disconnect(self.updatePlotList)
            #self.manager.removeWidget(self.name)
        #else:
            #print "no manager"

    def registerPlot(self, name):   ## for backward compatibility
        self.vb.register(name)
        #self.name = name
        #win = str(self.window())
        ##print "register", name, win
        #if win not in PlotItem.managers:
            #PlotItem.managers[win] = PlotWidgetManager()
        #self.manager = PlotItem.managers[win]
        #self.manager.addWidget(self, name)
        ##QtCore.QObject.connect(self.manager, QtCore.SIGNAL('widgetListChanged'), self.updatePlotList)
        #self.manager.sigWidgetListChanged.connect(self.updatePlotList)
        #self.updatePlotList()

    #def updatePlotList(self):
        #"""Update the list of all plotWidgets in the "link" combos"""
        ##print "update plot list", self
        #try:
            #for sc in [self.ctrl.xLinkCombo, self.ctrl.yLinkCombo]:
                #current = unicode(sc.currentText())
                #sc.blockSignals(True)
                #try:
                    #sc.clear()
                    #sc.addItem("")
                    #if self.manager is not None:
                        #for w in self.manager.listWidgets():
                            ##print w
                            #if w == self.name:
                                #continue
                            #sc.addItem(w)
                            #if w == current:
                                #sc.setCurrentIndex(sc.count()-1)
                #finally:
                    #sc.blockSignals(False)
                    #if unicode(sc.currentText()) != current:
                        #sc.currentItemChanged.emit()
        #except:
            #import gc
            #refs= gc.get_referrers(self)
            #print "  error during update of", self
            #print "  Referrers are:", refs
            #raise
        
        
        
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





    #def viewRangeChanged(self, vb, range):
        ##self.emit(QtCore.SIGNAL('viewChanged'), *args)
        #self.sigRangeChanged.emit(self, range)

    #def blockLink(self, b):
        #self.linksBlocked = b

    #def xLinkComboChanged(self):
        #self.setXLink(str(self.ctrl.xLinkCombo.currentText()))

    #def yLinkComboChanged(self):
        #self.setYLink(str(self.ctrl.yLinkCombo.currentText()))

    #def setXLink(self, plot=None):
        #"""Link this plot's X axis to another plot (pass either the PlotItem/PlotWidget or the registered name of the plot)"""
        #if isinstance(plot, basestring):
            #if self.manager is None:
                #return
            #if self.xLinkPlot is not None:
                #self.manager.unlinkX(self, self.xLinkPlot)
            #plot = self.manager.getWidget(plot)
        #if not isinstance(plot, PlotItem) and hasattr(plot, 'getPlotItem'):
            #plot = plot.getPlotItem()
        #self.xLinkPlot = plot
        #if plot is not None:
            #self.setManualXScale()
            #self.manager.linkX(self, plot)

            

    #def setYLink(self, plot=None):
        #"""Link this plot's Y axis to another plot (pass either the PlotItem/PlotWidget or the registered name of the plot)"""
        #if isinstance(plot, basestring):
            #if self.manager is None:
                #return
            #if self.yLinkPlot is not None:
                #self.manager.unlinkY(self, self.yLinkPlot)
            #plot = self.manager.getWidget(plot)
        #if not isinstance(plot, PlotItem) and hasattr(plot, 'getPlotItem'):
            #plot = plot.getPlotItem()
        #self.yLinkPlot = plot
        #if plot is not None:
            #self.setManualYScale()
            #self.manager.linkY(self, plot)
        
    #def linkXChanged(self, plot):
        #"""Called when a linked plot has changed its X scale"""
        ##print "update from", plot
        #if self.linksBlocked:
            #return
        #pr = plot.vb.viewRect()
        #pg = plot.viewGeometry()
        #if pg is None:
            ##print "   return early"
            #return
        #sg = self.viewGeometry()
        #upp = float(pr.width()) / pg.width()
        #x1 = pr.left() + (sg.x()-pg.x()) * upp
        #x2 = x1 + sg.width() * upp
        #plot.blockLink(True)
        #self.setManualXScale()
        #self.setXRange(x1, x2, padding=0)
        #plot.blockLink(False)
        #self.replot()
        
    #def linkYChanged(self, plot):
        #"""Called when a linked plot has changed its Y scale"""
        #if self.linksBlocked:
            #return
        #pr = plot.vb.viewRect()
        #pg = plot.vb.boundingRect()
        #sg = self.vb.boundingRect()
        #upp = float(pr.height()) / pg.height()
        #y1 = pr.bottom() + (sg.y()-pg.y()) * upp
        #y2 = y1 + sg.height() * upp
        #plot.blockLink(True)
        #self.setManualYScale()
        #self.setYRange(y1, y2, padding=0)
        #plot.blockLink(False)
        #self.replot()


    def avgToggled(self, b):
        if b:
            self.recomputeAverages()
        for k in self.avgCurves:
            self.avgCurves[k][1].setVisible(b)
        
    def avgParamListClicked(self, item):
        name = str(item.text())
        self.paramList[name] = (item.checkState() == QtCore.Qt.Checked)
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
                if item.checkState() == QtCore.Qt.Checked:
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
            plot.setPen(fn.mkPen([0, 200, 0]))
            plot.setShadowPen(fn.mkPen([0, 0, 0, 100], width=3))
            plot.setAlpha(1.0, False)
            plot.setZValue(100)
            self.addItem(plot, skipAverage=True)
            self.avgCurves[key] = [0, plot]
        self.avgCurves[key][0] += 1
        (n, plot) = self.avgCurves[key]
        
        ### Average data together
        (x, y) = curve.getData()
        if plot.yData is not None:
            newData = plot.yData * (n-1) / float(n) + y * 1.0 / float(n)
            plot.setData(plot.xData, newData)
        else:
            plot.setData(x, y)
        

    #def mouseCheckChanged(self):
        #state = [self.ctrl.xMouseCheck.isChecked(), self.ctrl.yMouseCheck.isChecked()]
        #self.vb.setMouseEnabled(*state)
        
    #def xRangeChanged(self, _, range):
        #if any(np.isnan(range)) or any(np.isinf(range)):
            #raise Exception("yRange invalid: %s. Signal came from %s" % (str(range), str(self.sender())))
        #self.ctrl.xMinText.setText('%0.5g' % range[0])
        #self.ctrl.xMaxText.setText('%0.5g' % range[1])
        
        ### automatically change unit scale
        #maxVal = max(abs(range[0]), abs(range[1]))
        #(scale, prefix) = fn.siScale(maxVal)
        ##for l in ['top', 'bottom']:
            ##if self.getLabel(l).isVisible():
                ##self.setLabel(l, unitPrefix=prefix)
                ##self.getScale(l).setScale(scale)
            ##else:
                ##self.setLabel(l, unitPrefix='')
                ##self.getScale(l).setScale(1.0)
        
        ##self.emit(QtCore.SIGNAL('xRangeChanged'), self, range)
        #self.sigXRangeChanged.emit(self, range)

    #def yRangeChanged(self, _, range):
        #if any(np.isnan(range)) or any(np.isinf(range)):
            #raise Exception("yRange invalid: %s. Signal came from %s" % (str(range), str(self.sender())))
        #self.ctrl.yMinText.setText('%0.5g' % range[0])
        #self.ctrl.yMaxText.setText('%0.5g' % range[1])
        
        ### automatically change unit scale
        #maxVal = max(abs(range[0]), abs(range[1]))
        #(scale, prefix) = fn.siScale(maxVal)
        ##for l in ['left', 'right']:
            ##if self.getLabel(l).isVisible():
                ##self.setLabel(l, unitPrefix=prefix)
                ##self.getScale(l).setScale(scale)
            ##else:
                ##self.setLabel(l, unitPrefix='')
                ##self.getScale(l).setScale(1.0)
        ##self.emit(QtCore.SIGNAL('yRangeChanged'), self, range)
        #self.sigYRangeChanged.emit(self, range)

    def autoBtnClicked(self):
        if self.autoBtn.mode == 'auto':
            self.enableAutoRange()
        else:
            self.disableAutoRange()
            
    def enableAutoScale(self):
        """
        Enable auto-scaling. The plot will continuously scale to fit the boundaries of its data.
        """
        print("Warning: enableAutoScale is deprecated. Use enableAutoRange(axis, enable) instead.")
        self.vb.enableAutoRange(self.vb.XYAxes)
        #self.ctrl.xAutoRadio.setChecked(True)
        #self.ctrl.yAutoRadio.setChecked(True)
        
        #self.autoBtn.setImageFile(self.lockImageFile)
        #self.autoBtn.mode = 'lock'
        #self.updateXScale()
        #self.updateYScale()
        #self.replot()
      
    #def updateXScale(self):
        #"""Set plot to autoscale or not depending on state of radio buttons"""
        #if self.ctrl.xManualRadio.isChecked():
            #self.setManualXScale()
        #else:
            #self.setAutoXScale()
        #self.replot()
        
    #def updateYScale(self, b=False):
        #"""Set plot to autoscale or not depending on state of radio buttons"""
        #if self.ctrl.yManualRadio.isChecked():
            #self.setManualYScale()
        #else:
            #self.setAutoYScale()
        #self.replot()

    #def enableManualScale(self, v=[True, True]):
        #if v[0]:
            #self.autoScale[0] = False
            #self.ctrl.xManualRadio.setChecked(True)
            ##self.setManualXScale()
        #if v[1]:
            #self.autoScale[1] = False
            #self.ctrl.yManualRadio.setChecked(True)
            ##self.setManualYScale()
        ##self.autoBtn.enable()
        #self.autoBtn.setImageFile(self.autoImageFile)
        #self.autoBtn.mode = 'auto'
        ##self.replot()
        
    #def setManualXScale(self):
        #self.autoScale[0] = False
        #x1 = float(self.ctrl.xMinText.text())
        #x2 = float(self.ctrl.xMaxText.text())
        #self.ctrl.xManualRadio.setChecked(True)
        #self.setXRange(x1, x2, padding=0)
        #self.autoBtn.show()
        ##self.replot()
        
    #def setManualYScale(self):
        #self.autoScale[1] = False
        #y1 = float(self.ctrl.yMinText.text())
        #y2 = float(self.ctrl.yMaxText.text())
        #self.ctrl.yManualRadio.setChecked(True)
        #self.setYRange(y1, y2, padding=0)
        #self.autoBtn.show()
        ##self.replot()

    #def setAutoXScale(self):
        #self.autoScale[0] = True
        #self.ctrl.xAutoRadio.setChecked(True)
        ##self.replot()
        
    #def setAutoYScale(self):
        #self.autoScale[1] = True
        #self.ctrl.yAutoRadio.setChecked(True)
        ##self.replot()

    def addItem(self, item, *args, **kargs):
        """
        Add a graphics item to the view box. 
        If the item has plot data (PlotDataItem, PlotCurveItem, ScatterPlotItem), it may
        be included in analysis performed by the PlotItem.
        """
        self.items.append(item)
        vbargs = {}
        if 'ignoreBounds' in kargs:
            vbargs['ignoreBounds'] = kargs['ignoreBounds']
        self.vb.addItem(item, *args, **vbargs)
        if hasattr(item, 'implements') and item.implements('plotData'):
            self.dataItems.append(item)
            #self.plotChanged()
            
            params = kargs.get('params', {})
            self.itemMeta[item] = params
            #item.setMeta(params)
            self.curves.append(item)
            #self.addItem(c)
            
        if isinstance(item, PlotDataItem):
            ## configure curve for this plot
            (alpha, auto) = self.alphaState()
            item.setAlpha(alpha, auto)
            item.setFftMode(self.ctrl.fftCheck.isChecked())
            item.setLogMode(self.ctrl.logXCheck.isChecked(), self.ctrl.logYCheck.isChecked())
            item.setDownsampling(self.downsampleMode())
            item.setPointMode(self.pointMode())
            
            ## Hide older plots if needed
            self.updateDecimation()
            
            ## Add to average if needed
            self.updateParamList()
            if self.ctrl.averageGroup.isChecked() and 'skipAverage' not in kargs:
                self.addAvgCurve(item)
                
            #c.connect(c, QtCore.SIGNAL('plotChanged'), self.plotChanged)
            #item.sigPlotChanged.connect(self.plotChanged)
            #self.plotChanged()

    def addDataItem(self, item, *args):
        print("PlotItem.addDataItem is deprecated. Use addItem instead.")
        self.addItem(item, *args)
        
    def addCurve(self, c, params=None):
        print("PlotItem.addCurve is deprecated. Use addItem instead.")
        self.addItem(c, params)

    def removeItem(self, item):
        """
        Remove an item from the internal ViewBox.
        """
        if not item in self.items:
            return
        self.items.remove(item)
        if item in self.dataItems:
            self.dataItems.remove(item)
            
        if item.scene() is not None:
            self.vb.removeItem(item)
        if item in self.curves:
            self.curves.remove(item)
            self.updateDecimation()
            self.updateParamList()
            #item.connect(item, QtCore.SIGNAL('plotChanged'), self.plotChanged)
            #item.sigPlotChanged.connect(self.plotChanged)

    def clear(self):
        """
        Remove all items from the ViewBox.
        """
        for i in self.items[:]:
            self.removeItem(i)
        self.avgCurves = {}
    
    def clearPlots(self):
        for i in self.curves[:]:
            self.removeItem(i)
        self.avgCurves = {}
        
    
    def plot(self, *args, **kargs):
        """
        Add and return a new plot.
        See :func:`PlotDataItem.__init__ <pyqtgraph.PlotDataItem.__init__>` for data arguments
        
        Extra allowed arguments are:
            clear    - clear all plots before displaying new data
            params   - meta-parameters to associate with this data
        """
        
        
        
        #if y is not None:
            #data = y
        #if data2 is not None:
            #x = data
            #data = data2
        #if decimate is not None and decimate > 1:
            #data = data[::decimate]
            #if x is not None:
                #x = x[::decimate]
          ##  print 'plot with decimate = %d' % (decimate)
        clear = kargs.get('clear', False)
        params = kargs.get('params', None)
          
        if clear:
            self.clear()
            
        item = PlotDataItem(*args, **kargs)
            
        if params is None:
            params = {}
        #if HAVE_METAARRAY and isinstance(data, MetaArray):
            #curve = self._plotMetaArray(data, x=x, **kargs)
        #elif isinstance(data, np.ndarray):
            #curve = self._plotArray(data, x=x, **kargs)
        #elif isinstance(data, list):
            #if x is not None:
                #x = np.array(x)
            #curve = self._plotArray(np.array(data), x=x, **kargs)
        #elif data is None:
            #curve = PlotCurveItem(**kargs)
        #else:
            #raise Exception('Not sure how to plot object of type %s' % type(data))
            
        #print data, curve
        self.addItem(item, params=params)
        #if pen is not None:
            #curve.setPen(fn.mkPen(pen))
        
        return item

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
        #sp = ScatterPlotItem(*args, **kargs)
        #self.addItem(sp)
        #return sp

    

    #def plotChanged(self, curve=None):
        ## Recompute auto range if needed
        #args = {}
        #for ax in [0, 1]:
            #print "range", ax
            #if self.autoScale[ax]:
                #percentScale = [self.ctrl.xAutoPercentSpin.value(), self.ctrl.yAutoPercentSpin.value()][ax] * 0.01
                #mn = None
                #mx = None
                #for c in self.curves + [c[1] for c in self.avgCurves.values()] + self.dataItems:
                    #if not c.isVisible():
                        #continue
                    #cmn, cmx = c.getRange(ax, percentScale)
                    ##print "   ", c, cmn, cmx
                    #if mn is None or cmn < mn:
                        #mn = cmn
                    #if mx is None or cmx > mx:
                        #mx = cmx
                #if mn is None or mx is None or any(np.isnan([mn, mx])) or any(np.isinf([mn, mx])):
                    #continue
                #if mn == mx:
                    #mn -= 1
                    #mx += 1
                #if ax == 0:
                    #args['xRange'] = [mn, mx]
                #else:
                    #args['yRange'] = [mn, mx]
                    
        #if len(args) > 0:
            ##print args
            #self.setRange(**args)
                
    def replot(self):
        #self.plotChanged()
        self.update()

    def updateParamList(self):
        self.ctrl.avgParamList.clear()
        ## Check to see that each parameter for each curve is present in the list
        #print "\nUpdate param list", self
        #print "paramList:", self.paramList
        for c in self.curves:
            #print "  curve:", c
            for p in list(self.itemMeta.get(c, {}).keys()):
                #print "    param:", p
                if type(p) is tuple:
                    p = '.'.join(p)
                    
                ## If the parameter is not in the list, add it.
                matches = self.ctrl.avgParamList.findItems(p, QtCore.Qt.MatchExactly)
                #print "      matches:", matches
                if len(matches) == 0:
                    i = QtGui.QListWidgetItem(p)
                    if p in self.paramList and self.paramList[p] is True:
                        #print "      set checked"
                        i.setCheckState(QtCore.Qt.Checked)
                    else:
                        #print "      set unchecked"
                        i.setCheckState(QtCore.Qt.Unchecked)
                    self.ctrl.avgParamList.addItem(i)
                else:
                    i = matches[0]
                    
                self.paramList[p] = (i.checkState() == QtCore.Qt.Checked)
        #print "paramList:", self.paramList


    ## This is bullshit.
    def writeSvgCurves(self, fileName=None):
        if fileName is None:
            self.fileDialog = FileDialog()
            if PlotItem.lastFileDir is not None:
                self.fileDialog.setDirectory(PlotItem.lastFileDir)
            self.fileDialog.setFileMode(QtGui.QFileDialog.AnyFile)
            self.fileDialog.setAcceptMode(QtGui.QFileDialog.AcceptSave) 
            self.fileDialog.show()
            self.fileDialog.fileSelected.connect(self.writeSvg)
            return
        #if fileName is None:
            #fileName = QtGui.QFileDialog.getSaveFileName()
        if isinstance(fileName, tuple):
            raise Exception("Not implemented yet..")
        fileName = str(fileName)
        PlotItem.lastFileDir = os.path.dirname(fileName)
        
        rect = self.vb.viewRect()
        xRange = rect.left(), rect.right() 
        
        svg = ""
        fh = open(fileName, 'w')

        dx = max(rect.right(),0) - min(rect.left(),0)
        ymn = min(rect.top(), rect.bottom())
        ymx = max(rect.top(), rect.bottom())
        dy = max(ymx,0) - min(ymn,0)
        sx = 1.
        sy = 1.
        while dx*sx < 10:
            sx *= 1000
        while dy*sy < 10:
            sy *= 1000
        sy *= -1

        #fh.write('<svg viewBox="%f %f %f %f">\n' % (rect.left()*sx, rect.top()*sx, rect.width()*sy, rect.height()*sy))
        fh.write('<svg>\n')
        fh.write('<path fill="none" stroke="#000000" stroke-opacity="0.5" stroke-width="1" d="M%f,0 L%f,0"/>\n' % (rect.left()*sx, rect.right()*sx))
        fh.write('<path fill="none" stroke="#000000" stroke-opacity="0.5" stroke-width="1" d="M0,%f L0,%f"/>\n' % (rect.top()*sy, rect.bottom()*sy))


        for item in self.curves:
            if isinstance(item, PlotCurveItem):
                color = fn.colorStr(item.pen.color())
                opacity = item.pen.color().alpha() / 255.
                color = color[:6]
                x, y = item.getData()
                mask = (x > xRange[0]) * (x < xRange[1])
                mask[:-1] += mask[1:]
                m2 = mask.copy()
                mask[1:] += m2[:-1]
                x = x[mask]
                y = y[mask]
                
                x *= sx
                y *= sy
                
                #fh.write('<g fill="none" stroke="#%s" stroke-opacity="1" stroke-width="1">\n' % color)
                fh.write('<path fill="none" stroke="#%s" stroke-opacity="%f" stroke-width="1" d="M%f,%f ' % (color, opacity, x[0], y[0]))
                for i in range(1, len(x)):
                    fh.write('L%f,%f ' % (x[i], y[i]))
                
                fh.write('"/>')
                #fh.write("</g>")
        for item in self.dataItems:
            if isinstance(item, ScatterPlotItem):
                
                pRect = item.boundingRect()
                vRect = pRect.intersected(rect)
                
                for point in item.points():
                    pos = point.pos()
                    if not rect.contains(pos):
                        continue
                    color = fn.colorStr(point.brush.color())
                    opacity = point.brush.color().alpha() / 255.
                    color = color[:6]
                    x = pos.x() * sx
                    y = pos.y() * sy
                    
                    fh.write('<circle cx="%f" cy="%f" r="1" fill="#%s" stroke="none" fill-opacity="%f"/>\n' % (x, y, color, opacity))
                    #fh.write('<path fill="none" stroke="#%s" stroke-opacity="%f" stroke-width="1" d="M%f,%f ' % (color, opacity, x[0], y[0]))
                    #for i in xrange(1, len(x)):
                        #fh.write('L%f,%f ' % (x[i], y[i]))
                    
                    #fh.write('"/>')
            
        ## get list of curves, scatter plots
        
        
        fh.write("</svg>\n")
        
        
    
    def writeSvg(self, fileName=None):
        if fileName is None:
            fileName = QtGui.QFileDialog.getSaveFileName()
        fileName = str(fileName)
        PlotItem.lastFileDir = os.path.dirname(fileName)
        
        self.svg = QtSvg.QSvgGenerator()
        self.svg.setFileName(fileName)
        res = 120.
        view = self.scene().views()[0]
        bounds = view.viewport().rect()
        bounds = QtCore.QRectF(0, 0, bounds.width(), bounds.height())
        
        self.svg.setResolution(res)
        self.svg.setViewBox(bounds)
        
        self.svg.setSize(QtCore.QSize(bounds.width(), bounds.height()))
        
        painter = QtGui.QPainter(self.svg)
        view.render(painter, bounds)
        
        painter.end()
        
        ## Workaround to set pen widths correctly
        import re
        data = open(fileName).readlines()
        for i in range(len(data)):
            line = data[i]
            m = re.match(r'(<g .*)stroke-width="1"(.*transform="matrix\(([^\)]+)\)".*)', line)
            if m is not None:
                #print "Matched group:", line
                g = m.groups()
                matrix = list(map(float, g[2].split(',')))
                #print "matrix:", matrix
                scale = max(abs(matrix[0]), abs(matrix[3]))
                if scale == 0 or scale == 1.0:
                    continue
                data[i] = g[0] + ' stroke-width="%0.2g" ' % (1.0/scale) + g[1] + '\n'
                #print "old line:", line
                #print "new line:", data[i]
        open(fileName, 'w').write(''.join(data))
        
        
    def writeImage(self, fileName=None):
        if fileName is None:
            self.fileDialog = FileDialog()
            if PlotItem.lastFileDir is not None:
                self.fileDialog.setDirectory(PlotItem.lastFileDir)
            self.fileDialog.setFileMode(QtGui.QFileDialog.AnyFile)
            self.fileDialog.setAcceptMode(QtGui.QFileDialog.AcceptSave) 
            self.fileDialog.show()
            self.fileDialog.fileSelected.connect(self.writeImage)
            return
        #if fileName is None:
            #fileName = QtGui.QFileDialog.getSaveFileName()
        if isinstance(fileName, tuple):
            raise Exception("Not implemented yet..")
        fileName = str(fileName)
        PlotItem.lastFileDir = os.path.dirname(fileName)
        self.png = QtGui.QImage(int(self.size().width()), int(self.size().height()), QtGui.QImage.Format_ARGB32)
        painter = QtGui.QPainter(self.png)
        painter.setRenderHints(painter.Antialiasing | painter.TextAntialiasing)
        self.scene().render(painter, QtCore.QRectF(), self.mapRectToScene(self.boundingRect()))
        painter.end()
        self.png.save(fileName)
        
    def writeCsv(self, fileName=None):
        if fileName is None:
            self.fileDialog = FileDialog()
            if PlotItem.lastFileDir is not None:
                self.fileDialog.setDirectory(PlotItem.lastFileDir)
            self.fileDialog.setFileMode(QtGui.QFileDialog.AnyFile)
            self.fileDialog.setAcceptMode(QtGui.QFileDialog.AcceptSave) 
            self.fileDialog.show()
            self.fileDialog.fileSelected.connect(self.writeCsv)
            return
        #if fileName is None:
            #fileName = QtGui.QFileDialog.getSaveFileName()
        fileName = str(fileName)
        PlotItem.lastFileDir = os.path.dirname(fileName)
        
        fd = open(fileName, 'w')
        data = [c.getData() for c in self.curves]
        i = 0
        while True:
            done = True
            for d in data:
                if i < len(d[0]):
                    fd.write('%g,%g,'%(d[0][i], d[1][i]))
                    done = False
                else:
                    fd.write(' , ,')
            fd.write('\n')
            if done:
                break
            i += 1
        fd.close()


    def saveState(self):
        #if not HAVE_WIDGETGROUP:
            #raise Exception("State save/restore requires WidgetGroup class.")
        state = self.stateGroup.state()
        state['paramList'] = self.paramList.copy()
        state['view'] = self.vb.getState()
        #print "\nSAVE %s:\n" % str(self.name), state
        #print "Saving state. averageGroup.isChecked(): %s  state: %s" % (str(self.ctrl.averageGroup.isChecked()), str(state['averageGroup']))
        return state
        
    def restoreState(self, state):
        #if not HAVE_WIDGETGROUP:
            #raise Exception("State save/restore requires WidgetGroup class.")
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
        #self.updateXScale()
        #self.updateYScale()
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
        
        
        #print "\nRESTORE %s:\n" % str(self.name), state
        #print "Restoring state. averageGroup.isChecked(): %s  state: %s" % (str(self.ctrl.averageGroup.isChecked()), str(state['averageGroup']))
        #avg = self.ctrl.averageGroup.isChecked()
        #if avg != state['averageGroup']:
            #print "  WARNING: avgGroup is %s, should be %s" % (str(avg), str(state['averageGroup']))


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
        for c in self.curves:
            c.setLogMode(x,y)
        self.getAxis('bottom').setLogMode(x)
        self.getAxis('top').setLogMode(x)
        self.getAxis('left').setLogMode(y)
        self.getAxis('right').setLogMode(y)
        self.enableAutoRange()
        self.recomputeAverages()
        
        
    def updateDownsampling(self):
        ds = self.downsampleMode()
        for c in self.curves:
            c.setDownsampling(ds)
        self.recomputeAverages()
        #for c in self.avgCurves.values():
            #c[1].setDownsampling(ds)
        
        
    def downsampleMode(self):
        if self.ctrl.decimateGroup.isChecked():
            if self.ctrl.manualDecimateRadio.isChecked():
                ds = self.ctrl.downsampleSpin.value()
            else:
                ds = True
        else:
            ds = False
        return ds
        
    def updateDecimation(self):
        if self.ctrl.maxTracesCheck.isChecked():
            numCurves = self.ctrl.maxTracesSpin.value()
        else:
            numCurves = -1
            
        curves = self.curves[:]
        split = len(curves) - numCurves
        for i in range(len(curves)):
            if numCurves == -1 or i >= split:
                curves[i].show()
            else:
                if self.ctrl.forgetTracesCheck.isChecked():
                    curves[i].clear()
                    self.removeItem(curves[i])
                else:
                    curves[i].hide()
        
      
    def updateAlpha(self, *args):
        (alpha, auto) = self.alphaState()
        for c in self.curves:
            c.setAlpha(alpha**2, auto)
                
        #self.replot(autoRange=False)
     
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
        
    #def wheelEvent(self, ev):
        ## disables default panning the whole scene by mousewheel
        #ev.accept()

    def resizeEvent(self, ev):
        if self.autoBtn is None:  ## already closed down
            return
        btnRect = self.mapRectFromItem(self.autoBtn, self.autoBtn.boundingRect())
        y = self.size().height() - btnRect.height()
        self.autoBtn.setPos(0, y)
        
    #def hoverMoveEvent(self, ev):
        #self.mousePos = ev.pos()
        #self.mouseScreenPos = ev.screenPos()
        
        
    #def ctrlBtnClicked(self):
        #self.ctrlMenu.popup(self.mouseScreenPos)
        
    def getMenu(self):
        return self.ctrlMenu
    
    def getContextMenus(self, event):
        ## called when another item is displaying its context menu; we get to add extras to the end of the menu.
        return self.ctrlMenu
        

    def getLabel(self, key):
        pass
        
    def _checkScaleKey(self, key):
        if key not in self.scales:
            raise Exception("Scale '%s' not found. Scales are: %s" % (key, str(list(self.scales.keys()))))
        
    def getScale(self, key):
        return self.getAxis(key)
        
    def getAxis(self, name):
        """Return the specified AxisItem. 
        *name* should be 'left', 'bottom', 'top', or 'right'."""
        self._checkScaleKey(name)
        return self.scales[name]['item']
        
    def setLabel(self, axis, text=None, units=None, unitPrefix=None, **args):
        """
        Set the label for an axis. Basic HTML formatting is allowed.
        
        ============= =================================================================
        **Arguments**
        axis          must be one of 'left', 'bottom', 'right', or 'top'
        text          text to display along the axis. HTML allowed.
        units         units to display after the title. If units are given, 
                      then an SI prefix will be automatically appended
                      and the axis values will be scaled accordingly.
                      (ie, use 'V' instead of 'mV'; 'm' will be added automatically)
        ============= =================================================================
        """
        self.getScale(axis).setLabel(text=text, units=units, **args)
        
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
        p = self.scales[axis]['pos']
        if show:
            s.show()
        else:
            s.hide()
            
    def hideAxis(self, axis):
        self.showAxis(axis, False)
            
    def showScale(self, *args, **kargs):
        print("Deprecated. use showAxis() instead")
        return self.showAxis(*args, **kargs)
            
    def hideButtons(self):
        #self.ctrlBtn.hide()
        self.autoBtn.hide()
        
            
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
        inf = arr.infoCopy()
        if arr.ndim != 1:
            raise Exception('can only automatically plot 1 dimensional arrays.')
        ## create curve
        try:
            xv = arr.xvals(0)
            #print 'xvals:', xv
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

    #def saveSvgClicked(self):
        #self.writeSvg()
        
    #def saveSvgCurvesClicked(self):
        #self.writeSvgCurves()
        
    #def saveImgClicked(self):
        #self.writeImage()
            
    #def saveCsvClicked(self):
        #self.writeCsv()
      
    def setExportMode(self, export, opts):
        if export:
            self.autoBtn.hide()
        else:
            self.autoBtn.show()
    

#class PlotWidgetManager(QtCore.QObject):
    
    #sigWidgetListChanged = QtCore.Signal(object)
    
    #"""Used for managing communication between PlotWidgets"""
    #def __init__(self):
        #QtCore.QObject.__init__(self)
        #self.widgets = weakref.WeakValueDictionary() # Don't keep PlotWidgets around just because they are listed here
    
    #def addWidget(self, w, name):
        #self.widgets[name] = w
        ##self.emit(QtCore.SIGNAL('widgetListChanged'), self.widgets.keys())
        #self.sigWidgetListChanged.emit(self.widgets.keys())
        
    #def removeWidget(self, name):
        #if name in self.widgets:
            #del self.widgets[name]
            ##self.emit(QtCore.SIGNAL('widgetListChanged'), self.widgets.keys())
            #self.sigWidgetListChanged.emit(self.widgets.keys())
        #else:
            #print "plot %s not managed" % name
        
        
    #def listWidgets(self):
        #return self.widgets.keys()
        
    #def getWidget(self, name):
        #if name not in self.widgets:
            #return None
        #else:
            #return self.widgets[name]
            
    #def linkX(self, p1, p2):
        ##QtCore.QObject.connect(p1, QtCore.SIGNAL('xRangeChanged'), p2.linkXChanged)
        #p1.sigXRangeChanged.connect(p2.linkXChanged)
        ##QtCore.QObject.connect(p2, QtCore.SIGNAL('xRangeChanged'), p1.linkXChanged)
        #p2.sigXRangeChanged.connect(p1.linkXChanged)
        #p1.linkXChanged(p2)
        ##p2.setManualXScale()

    #def unlinkX(self, p1, p2):
        ##QtCore.QObject.disconnect(p1, QtCore.SIGNAL('xRangeChanged'), p2.linkXChanged)
        #p1.sigXRangeChanged.disconnect(p2.linkXChanged)
        ##QtCore.QObject.disconnect(p2, QtCore.SIGNAL('xRangeChanged'), p1.linkXChanged)
        #p2.sigXRangeChanged.disconnect(p1.linkXChanged)
        
    #def linkY(self, p1, p2):
        ##QtCore.QObject.connect(p1, QtCore.SIGNAL('yRangeChanged'), p2.linkYChanged)
        #p1.sigYRangeChanged.connect(p2.linkYChanged)
        ##QtCore.QObject.connect(p2, QtCore.SIGNAL('yRangeChanged'), p1.linkYChanged)
        #p2.sigYRangeChanged.connect(p1.linkYChanged)
        #p1.linkYChanged(p2)
        ##p2.setManualYScale()

    #def unlinkY(self, p1, p2):
        ##QtCore.QObject.disconnect(p1, QtCore.SIGNAL('yRangeChanged'), p2.linkYChanged)
        #p1.sigYRangeChanged.disconnect(p2.linkYChanged)
        ##QtCore.QObject.disconnect(p2, QtCore.SIGNAL('yRangeChanged'), p1.linkYChanged)
        #p2.sigYRangeChanged.disconnect(p1.linkYChanged)
