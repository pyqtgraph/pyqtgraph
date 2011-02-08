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

from graphicsItems import *
from plotConfigTemplate import *
from PyQt4 import QtGui, QtCore, QtSvg
from functions import *
#from ObjectWorkaround import *
#tryWorkaround(QtCore, QtGui)
import weakref
import numpy as np

try:
    from WidgetGroup import *
    HAVE_WIDGETGROUP = True
except:
    HAVE_WIDGETGROUP = False
    
try:
    from metaarray import *
    HAVE_METAARRAY = True
except:
    HAVE_METAARRAY = False


class PlotItem(QtGui.QGraphicsWidget):
    """Plot graphics item that can be added to any graphics scene. Implements axis titles, scales, interactive viewbox."""
    lastFileDir = None
    managers = {}
    
    def __init__(self, parent=None, name=None, labels=None, **kargs):
        QtGui.QGraphicsWidget.__init__(self, parent)
        
        ## Set up control buttons
        
        self.ctrlBtn = QtGui.QToolButton()
        self.ctrlBtn.setText('?')
        self.autoBtn = QtGui.QToolButton()
        self.autoBtn.setText('A')
        self.autoBtn.hide()
        
        for b in [self.ctrlBtn, self.autoBtn]:
            proxy = QtGui.QGraphicsProxyWidget(self)
            proxy.setWidget(b)
            proxy.setAcceptHoverEvents(False)
            b.setStyleSheet("background-color: #000000; color: #888; font-size: 6pt")
        QtCore.QObject.connect(self.ctrlBtn, QtCore.SIGNAL('clicked()'), self.ctrlBtnClicked)
        QtCore.QObject.connect(self.autoBtn, QtCore.SIGNAL('clicked()'), self.enableAutoScale)
        
        
        self.layout = QtGui.QGraphicsGridLayout()
        self.layout.setContentsMargins(1,1,1,1)
        self.setLayout(self.layout)
        self.layout.setHorizontalSpacing(0)
        self.layout.setVerticalSpacing(0)
        
        self.vb = ViewBox()
        QtCore.QObject.connect(self.vb, QtCore.SIGNAL('xRangeChanged'), self.xRangeChanged)
        QtCore.QObject.connect(self.vb, QtCore.SIGNAL('yRangeChanged'), self.yRangeChanged)
        QtCore.QObject.connect(self.vb, QtCore.SIGNAL('rangeChangedManually'), self.enableManualScale)
        
        QtCore.QObject.connect(self.vb, QtCore.SIGNAL('viewChanged'), self.viewChanged)
        
        self.layout.addItem(self.vb, 2, 1)
        self.alpha = 1.0
        self.autoAlpha = True
        self.spectrumMode = False
         
        self.autoScale = [True, True]
         
        ## Create and place scale items
        self.scales = {
            'top':    {'item': ScaleItem(orientation='top',    linkView=self.vb), 'pos': (1, 1)}, 
            'bottom': {'item': ScaleItem(orientation='bottom', linkView=self.vb), 'pos': (3, 1)}, 
            'left':   {'item': ScaleItem(orientation='left',   linkView=self.vb), 'pos': (2, 0)}, 
            'right':  {'item': ScaleItem(orientation='right',  linkView=self.vb), 'pos': (2, 2)}
        }
        for k in self.scales:
            self.layout.addItem(self.scales[k]['item'], *self.scales[k]['pos'])
            
        ## Create and place label items
        #self.labels = {
            #'title':  {'item': LabelItem('title', size='11pt'),  'pos': (0, 2), 'text': ''},
            #'top':    {'item': LabelItem('top'),    'pos': (1, 2), 'text': '', 'units': '', 'unitPrefix': ''},
            #'bottom': {'item': LabelItem('bottom'), 'pos': (5, 2), 'text': '', 'units': '', 'unitPrefix': ''},
            #'left':   {'item': LabelItem('left'),   'pos': (3, 0), 'text': '', 'units': '', 'unitPrefix': ''},
            #'right':  {'item': LabelItem('right'),  'pos': (3, 4), 'text': '', 'units': '', 'unitPrefix': ''}
        #}
        #self.labels['left']['item'].setAngle(-90)
        #self.labels['right']['item'].setAngle(-90)
        #for k in self.labels:
            #self.layout.addItem(self.labels[k]['item'], *self.labels[k]['pos'])
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
        for m in ['setXRange', 'setYRange', 'setRange', 'autoRange', 'viewRect', 'setMouseEnabled']:
            setattr(self, m, getattr(self.vb, m))
            
        self.items = []
        self.curves = []
        self.dataItems = []
        self.paramList = {}
        self.avgCurves = {}
        
        ### Set up context menu
        
        w = QtGui.QWidget()
        self.ctrl = c = Ui_Form()
        c.setupUi(w)
        dv = QtGui.QDoubleValidator(self)
        self.ctrlMenu = QtGui.QMenu()
        ac = QtGui.QWidgetAction(self)
        ac.setDefaultWidget(w)
        self.ctrlMenu.addAction(ac)
        
        if HAVE_WIDGETGROUP:
            self.stateGroup = WidgetGroup(self.ctrlMenu)
        
        self.fileDialog = None
        
        self.xLinkPlot = None
        self.yLinkPlot = None
        self.linksBlocked = False

        
        #self.ctrlBtn.setFixedWidth(60)
        self.setAcceptHoverEvents(True)
        
        ## Connect control widgets
        QtCore.QObject.connect(c.xMinText, QtCore.SIGNAL('editingFinished()'), self.setManualXScale)
        QtCore.QObject.connect(c.xMaxText, QtCore.SIGNAL('editingFinished()'), self.setManualXScale)
        QtCore.QObject.connect(c.yMinText, QtCore.SIGNAL('editingFinished()'), self.setManualYScale)
        QtCore.QObject.connect(c.yMaxText, QtCore.SIGNAL('editingFinished()'), self.setManualYScale)
        
        QtCore.QObject.connect(c.xManualRadio, QtCore.SIGNAL('clicked()'), self.updateXScale)
        QtCore.QObject.connect(c.yManualRadio, QtCore.SIGNAL('clicked()'), self.updateYScale)
        
        QtCore.QObject.connect(c.xAutoRadio, QtCore.SIGNAL('clicked()'), self.updateXScale)
        QtCore.QObject.connect(c.yAutoRadio, QtCore.SIGNAL('clicked()'), self.updateYScale)

        QtCore.QObject.connect(c.xAutoPercentSpin, QtCore.SIGNAL('valueChanged(int)'), self.replot)
        QtCore.QObject.connect(c.yAutoPercentSpin, QtCore.SIGNAL('valueChanged(int)'), self.replot)
        
        #QtCore.QObject.connect(c.xLogCheck, QtCore.SIGNAL('toggled(bool)'), self.setXLog)
        #QtCore.QObject.connect(c.yLogCheck, QtCore.SIGNAL('toggled(bool)'), self.setYLog)

        QtCore.QObject.connect(c.alphaGroup, QtCore.SIGNAL('toggled(bool)'), self.updateAlpha)
        QtCore.QObject.connect(c.alphaSlider, QtCore.SIGNAL('valueChanged(int)'), self.updateAlpha)
        QtCore.QObject.connect(c.autoAlphaCheck, QtCore.SIGNAL('toggled(bool)'), self.updateAlpha)

        QtCore.QObject.connect(c.gridGroup, QtCore.SIGNAL('toggled(bool)'), self.updateGrid)
        QtCore.QObject.connect(c.gridAlphaSlider, QtCore.SIGNAL('valueChanged(int)'), self.updateGrid)

        QtCore.QObject.connect(c.powerSpectrumGroup, QtCore.SIGNAL('toggled(bool)'), self.updateSpectrumMode)
        QtCore.QObject.connect(c.saveSvgBtn, QtCore.SIGNAL('clicked()'), self.saveSvgClicked)
        QtCore.QObject.connect(c.saveImgBtn, QtCore.SIGNAL('clicked()'), self.saveImgClicked)
        QtCore.QObject.connect(c.saveCsvBtn, QtCore.SIGNAL('clicked()'), self.saveCsvClicked)
        
        #QtCore.QObject.connect(c.gridGroup, QtCore.SIGNAL('toggled(bool)'), self.updateGrid)
        #QtCore.QObject.connect(c.gridAlphaSlider, QtCore.SIGNAL('valueChanged(int)'), self.updateGrid)
        
        QtCore.QObject.connect(self.ctrl.xLinkCombo, QtCore.SIGNAL('currentIndexChanged(int)'), self.xLinkComboChanged)
        QtCore.QObject.connect(self.ctrl.yLinkCombo, QtCore.SIGNAL('currentIndexChanged(int)'), self.yLinkComboChanged)

        QtCore.QObject.connect(c.downsampleSpin, QtCore.SIGNAL('valueChanged(int)'), self.updateDownsampling)

        QtCore.QObject.connect(self.ctrl.avgParamList, QtCore.SIGNAL('itemClicked(QListWidgetItem*)'), self.avgParamListClicked)
        QtCore.QObject.connect(self.ctrl.averageGroup, QtCore.SIGNAL('toggled(bool)'), self.avgToggled)
        
        #QtCore.QObject.connect(self.ctrl.pointsGroup, QtCore.SIGNAL('toggled(bool)'), self.updatePointMode)
        #QtCore.QObject.connect(self.ctrl.autoPointsCheck, QtCore.SIGNAL('toggled(bool)'), self.updatePointMode)
        
        QtCore.QObject.connect(self.ctrl.maxTracesCheck, QtCore.SIGNAL('toggled(bool)'), self.updateDecimation)
        QtCore.QObject.connect(self.ctrl.maxTracesSpin, QtCore.SIGNAL('valueChanged(int)'), self.updateDecimation)
        QtCore.QObject.connect(c.xMouseCheck, QtCore.SIGNAL('toggled(bool)'), self.mouseCheckChanged)
        QtCore.QObject.connect(c.yMouseCheck, QtCore.SIGNAL('toggled(bool)'), self.mouseCheckChanged)

        self.xLinkPlot = None
        self.yLinkPlot = None
        self.linksBlocked = False
        self.manager = None
        
        #self.showLabel('right', False)
        #self.showLabel('top', False)
        #self.showLabel('title', False)
        #self.showLabel('left', False)
        #self.showLabel('bottom', False)
        self.showScale('right', False)
        self.showScale('top', False)
        self.showScale('left', True)
        self.showScale('bottom', True)
        
        if name is not None:
            self.registerPlot(name)
        
        if labels is not None:
            for k in labels:
                if isinstance(labels[k], basestring):
                    labels[k] = (labels[k],)
                self.setLabel(k, *labels[k])
        
        if len(kargs) > 0:
            self.plot(**kargs)
        
            
    def __del__(self):
        if self.manager is not None:
            self.manager.removeWidget(self.name)

    def registerPlot(self, name):
        self.name = name
        win = str(self.window())
        #print "register", name, win
        if win not in PlotItem.managers:
            PlotItem.managers[win] = PlotWidgetManager()
        self.manager = PlotItem.managers[win]
        self.manager.addWidget(self, name)
        QtCore.QObject.connect(self.manager, QtCore.SIGNAL('widgetListChanged'), self.updatePlotList)
        self.updatePlotList()

    def updatePlotList(self):
        """Update the list of all plotWidgets in the "link" combos"""
        #print "update plot list", self
        try:
            for sc in [self.ctrl.xLinkCombo, self.ctrl.yLinkCombo]:
                current = str(sc.currentText())
                sc.clear()
                sc.addItem("")
                if self.manager is not None:
                    for w in self.manager.listWidgets():
                        #print w
                        if w == self.name:
                            continue
                        sc.addItem(w)
        except:
            import gc
            refs= gc.get_referrers(self)
            print "  error during update. Referrers are:", refs
            raise
        
    def updateGrid(self, *args):
        g = self.ctrl.gridGroup.isChecked()
        if g:
            g = self.ctrl.gridAlphaSlider.value()
        for k in self.scales:
            self.scales[k]['item'].setGrid(g)

    def viewGeometry(self):
        """return the screen geometry of the viewbox"""
        v = self.scene().views()[0]
        b = self.vb.mapRectToScene(self.vb.boundingRect())
        wr = v.mapFromScene(b).boundingRect()
        pos = v.mapToGlobal(v.pos())
        wr.adjust(pos.x(), pos.y(), pos.x(), pos.y())
        return wr




    def viewChanged(self, *args):
        self.emit(QtCore.SIGNAL('viewChanged'), *args)

    def blockLink(self, b):
        self.linksBlocked = b

    def xLinkComboChanged(self):
        self.setXLink(str(self.ctrl.xLinkCombo.currentText()))

    def yLinkComboChanged(self):
        self.setYLink(str(self.ctrl.yLinkCombo.currentText()))

    def setXLink(self, plot=None):
        """Link this plot's X axis to another plot (pass either the PlotItem/PlotWidget or the registered name of the plot)"""
        if isinstance(plot, basestring):
            if self.manager is None:
                return
            if self.xLinkPlot is not None:
                self.manager.unlinkX(self, self.xLinkPlot)
            plot = self.manager.getWidget(plot)
        if not isinstance(plot, PlotItem) and hasattr(plot, 'getPlotItem'):
            plot = plot.getPlotItem()
        self.xLinkPlot = plot
        if plot is not None:
            self.setManualXScale()
            self.manager.linkX(self, plot)
            
    def setYLink(self, plot=None):
        """Link this plot's Y axis to another plot (pass either the PlotItem/PlotWidget or the registered name of the plot)"""
        if isinstance(plot, basestring):
            if self.manager is None:
                return
            if self.yLinkPlot is not None:
                self.manager.unlinkY(self, self.yLinkPlot)
            plot = self.manager.getWidget(plot)
        if not isinstance(plot, PlotItem) and hasattr(plot, 'getPlotItem'):
            plot = plot.getPlotItem()
        self.yLinkPlot = plot
        if plot is not None:
            self.setManualYScale()
            self.manager.linkY(self, plot)
        
    def linkXChanged(self, plot):
        """Called when a linked plot has changed its X scale"""
        #print "update from", plot
        if self.linksBlocked:
            return
        pr = plot.vb.viewRect()
        pg = plot.viewGeometry()
        if pg is None:
            #print "   return early"
            return
        sg = self.viewGeometry()
        upp = float(pr.width()) / pg.width()
        x1 = pr.left() + (sg.x()-pg.x()) * upp
        x2 = x1 + sg.width() * upp
        plot.blockLink(True)
        self.setManualXScale()
        self.setXRange(x1, x2, padding=0)
        plot.blockLink(False)
        self.replot()
        
    def linkYChanged(self, plot):
        """Called when a linked plot has changed its Y scale"""
        if self.linksBlocked:
            return
        pr = plot.vb.viewRect()
        pg = plot.vb.boundingRect()
        sg = self.vb.boundingRect()
        upp = float(pr.height()) / pg.height()
        y1 = pr.bottom() + (sg.y()-pg.y()) * upp
        y2 = y1 + sg.height() * upp
        plot.blockLink(True)
        self.setManualYScale()
        self.setYRange(y1, y2, padding=0)
        plot.blockLink(False)
        self.replot()

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
            #Qwt.QwtPlotCurve.detach(self.avgCurves[k][1])
        self.avgCurves = {}
        for c in self.curves:
            self.addAvgCurve(c)
        self.replot()
        
    def addAvgCurve(self, curve):
        """Add a single curve into the pool of curves averaged together"""
        
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
                
        p = curve.meta().copy()
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
            plot = PlotCurveItem()
            plot.setPen(mkPen([0, 200, 0]))
            plot.setShadowPen(mkPen([0, 0, 0, 100], 3))
            plot.setAlpha(1.0, False)
            plot.setZValue(100)
            self.addItem(plot)
            #Qwt.QwtPlotCurve.attach(plot, self)
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
        
        
    def mouseCheckChanged(self):
        state = [self.ctrl.xMouseCheck.isChecked(), self.ctrl.yMouseCheck.isChecked()]
        self.vb.setMouseEnabled(*state)
        
    def xRangeChanged(self, _, range):
        if any(np.isnan(range)) or any(np.isinf(range)):
            raise Exception("yRange invalid: %s. Signal came from %s" % (str(range), str(self.sender())))
        self.ctrl.xMinText.setText('%0.5g' % range[0])
        self.ctrl.xMaxText.setText('%0.5g' % range[1])
        
        ## automatically change unit scale
        maxVal = max(abs(range[0]), abs(range[1]))
        (scale, prefix) = siScale(maxVal)
        #for l in ['top', 'bottom']:
            #if self.getLabel(l).isVisible():
                #self.setLabel(l, unitPrefix=prefix)
                #self.getScale(l).setScale(scale)
            #else:
                #self.setLabel(l, unitPrefix='')
                #self.getScale(l).setScale(1.0)
        
        self.emit(QtCore.SIGNAL('xRangeChanged'), self, range)

    def yRangeChanged(self, _, range):
        if any(np.isnan(range)) or any(np.isinf(range)):
            raise Exception("yRange invalid: %s. Signal came from %s" % (str(range), str(self.sender())))
        self.ctrl.yMinText.setText('%0.5g' % range[0])
        self.ctrl.yMaxText.setText('%0.5g' % range[1])
        
        ## automatically change unit scale
        maxVal = max(abs(range[0]), abs(range[1]))
        (scale, prefix) = siScale(maxVal)
        #for l in ['left', 'right']:
            #if self.getLabel(l).isVisible():
                #self.setLabel(l, unitPrefix=prefix)
                #self.getScale(l).setScale(scale)
            #else:
                #self.setLabel(l, unitPrefix='')
                #self.getScale(l).setScale(1.0)
        self.emit(QtCore.SIGNAL('yRangeChanged'), self, range)


    def enableAutoScale(self):
        self.ctrl.xAutoRadio.setChecked(True)
        self.ctrl.yAutoRadio.setChecked(True)
        self.autoBtn.hide()
        self.updateXScale()
        self.updateYScale()
        self.replot()
      
    def updateXScale(self):
        """Set plot to autoscale or not depending on state of radio buttons"""
        if self.ctrl.xManualRadio.isChecked():
            self.setManualXScale()
        else:
            self.setAutoXScale()
        self.replot()
        
    def updateYScale(self, b=False):
        """Set plot to autoscale or not depending on state of radio buttons"""
        if self.ctrl.yManualRadio.isChecked():
            self.setManualYScale()
        else:
            self.setAutoYScale()
        self.replot()

    def enableManualScale(self, v=[True, True]):
        if v[0]:
            self.autoScale[0] = False
            self.ctrl.xManualRadio.setChecked(True)
            #self.setManualXScale()
        if v[1]:
            self.autoScale[1] = False
            self.ctrl.yManualRadio.setChecked(True)
            #self.setManualYScale()
        self.autoBtn.show()
        #self.replot()
        
    def setManualXScale(self):
        self.autoScale[0] = False
        x1 = float(self.ctrl.xMinText.text())
        x2 = float(self.ctrl.xMaxText.text())
        self.ctrl.xManualRadio.setChecked(True)
        self.setXRange(x1, x2, padding=0)
        self.autoBtn.show()
        #self.replot()
        
    def setManualYScale(self):
        self.autoScale[1] = False
        y1 = float(self.ctrl.yMinText.text())
        y2 = float(self.ctrl.yMaxText.text())
        self.ctrl.yManualRadio.setChecked(True)
        self.setYRange(y1, y2, padding=0)
        self.autoBtn.show()
        #self.replot()

    def setAutoXScale(self):
        self.autoScale[0] = True
        self.ctrl.xAutoRadio.setChecked(True)
        #self.replot()
        
    def setAutoYScale(self):
        self.autoScale[1] = True
        self.ctrl.yAutoRadio.setChecked(True)
        #self.replot()

    def addItem(self, item, *args):
        self.items.append(item)
        self.vb.addItem(item, *args)
        
    def removeItem(self, item):
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
            item.connect(item, QtCore.SIGNAL('plotChanged'), self.plotChanged)

    def clear(self):
        for i in self.items[:]:
            self.removeItem(i)
        self.avgCurves = {}
    
    def clearPlots(self):
        for i in self.curves[:]:
            self.removeItem(i)
        self.avgCurves = {}
        
    
    def plot(self, data=None, data2=None, x=None, y=None, clear=False, params=None, pen=None):
        """Add a new plot curve. Data may be specified a few ways:
        plot(yVals)   # x vals will be integers
        plot(xVals, yVals)
        plot(y=yVals, x=xVals)
        """
        if y is not None:
            data = y
        if data2 is not None:
            x = data
            data = data2
        
        if clear:
            self.clear()
        if params is None:
            params = {}
        if HAVE_METAARRAY and isinstance(data, MetaArray):
            curve = self._plotMetaArray(data, x=x)
        elif isinstance(data, np.ndarray):
            curve = self._plotArray(data, x=x)
        elif isinstance(data, list):
            if x is not None:
                x = np.array(x)
            curve = self._plotArray(np.array(data), x=x)
        elif data is None:
            curve = PlotCurveItem()
        else:
            raise Exception('Not sure how to plot object of type %s' % type(data))
            
        #print data, curve
        self.addCurve(curve, params)
        if pen is not None:
            curve.setPen(mkPen(pen))
        
        return curve

    def addDataItem(self, item):
        self.addItem(item)
        self.dataItems.append(item)
    
    def addCurve(self, c, params=None):
        if params is None:
            params = {}
        c.setMeta(params)
        self.curves.append(c)
        #Qwt.QwtPlotCurve.attach(c, self)
        self.addItem(c)
        
        ## configure curve for this plot
        (alpha, auto) = self.alphaState()
        c.setAlpha(alpha, auto)
        c.setSpectrumMode(self.ctrl.powerSpectrumGroup.isChecked())
        c.setDownsampling(self.downsampleMode())
        c.setPointMode(self.pointMode())
        
        ## Hide older plots if needed
        self.updateDecimation()
        
        ## Add to average if needed
        self.updateParamList()
        if self.ctrl.averageGroup.isChecked():
            self.addAvgCurve(c)
            
        c.connect(c, QtCore.SIGNAL('plotChanged'), self.plotChanged)
        self.plotChanged()

    def plotChanged(self, curve=None):
        ## Recompute auto range if needed
        for ax in [0, 1]:
            if self.autoScale[ax]:
                percentScale = [self.ctrl.xAutoPercentSpin.value(), self.ctrl.yAutoPercentSpin.value()][ax] * 0.01
                mn = None
                mx = None
                for c in self.curves + [c[1] for c in self.avgCurves.values()] + self.dataItems:
                    if not c.isVisible():
                        continue
                    cmn, cmx = c.getRange(ax, percentScale)
                    if mn is None or cmn < mn:
                        mn = cmn
                    if mx is None or cmx > mx:
                        mx = cmx
                if mn is None or mx is None or any(np.isnan([mn, mx])) or any(np.isinf([mn, mx])):
                    continue
                if mn == mx:
                    mn -= 1
                    mx += 1
                self.setRange(ax, mn, mx)
                #print "Auto range:", ax, mn, mx
                
    def replot(self):
        self.plotChanged()
        self.update()

    def updateParamList(self):
        self.ctrl.avgParamList.clear()
        ## Check to see that each parameter for each curve is present in the list
        #print "\nUpdate param list", self
        #print "paramList:", self.paramList
        for c in self.curves:
            #print "  curve:", c
            for p in c.meta().keys():
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

    def writeSvg(self, fileName=None):
        if fileName is None:
            fileName = QtGui.QFileDialog.getSaveFileName()
        fileName = str(fileName)
        PlotItem.lastFileDir = os.path.dirname(fileName)
        
        self.svg = QtSvg.QSvgGenerator()
        self.svg.setFileName(fileName)
        res = 120.
        #bounds = self.mapRectToScene(self.boundingRect())
        view = self.scene().views()[0]
        bounds = view.viewport().rect()
        bounds = QtCore.QRectF(0, 0, bounds.width(), bounds.height())
        
        self.svg.setResolution(res)
        #self.svg.setSize(QtCore.QSize(self.size().width(), self.size().height()))
        self.svg.setViewBox(bounds)
        
        self.svg.setSize(QtCore.QSize(bounds.width(), bounds.height()))
        
        painter = QtGui.QPainter(self.svg)
        #self.scene().render(painter, QtCore.QRectF(), view.mapToScene(bounds).boundingRect())
        
        #items = self.scene().items()
        #self.scene().views()[0].drawItems(painter, len(items), items)
        
        #print view, type(view)
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
                matrix = map(float, g[2].split(','))
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
            fileName = QtGui.QFileDialog.getSaveFileName()
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
            fileName = QtGui.QFileDialog.getSaveFileName()
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
        if not HAVE_WIDGETGROUP:
            raise Exception("State save/restore requires WidgetGroup class.")
        state = self.stateGroup.state()
        state['paramList'] = self.paramList.copy()
        #print "\nSAVE %s:\n" % str(self.name), state
        #print "Saving state. averageGroup.isChecked(): %s  state: %s" % (str(self.ctrl.averageGroup.isChecked()), str(state['averageGroup']))
        return state
        
    def restoreState(self, state):
        if not HAVE_WIDGETGROUP:
            raise Exception("State save/restore requires WidgetGroup class.")
        if 'paramList' in state:
            self.paramList = state['paramList'].copy()
            
        self.stateGroup.setState(state)
        self.updateSpectrumMode()
        self.updateDownsampling()
        self.updateAlpha()
        self.updateDecimation()
        
        self.stateGroup.setState(state)
        self.updateXScale()
        self.updateYScale()
        self.updateParamList()
        
        #print "\nRESTORE %s:\n" % str(self.name), state
        #print "Restoring state. averageGroup.isChecked(): %s  state: %s" % (str(self.ctrl.averageGroup.isChecked()), str(state['averageGroup']))
        #avg = self.ctrl.averageGroup.isChecked()
        #if avg != state['averageGroup']:
            #print "  WARNING: avgGroup is %s, should be %s" % (str(avg), str(state['averageGroup']))


    def widgetGroupInterface(self):
        return (None, PlotItem.saveState, PlotItem.restoreState)
      
    def updateSpectrumMode(self, b=None):
        if b is None:
            b = self.ctrl.powerSpectrumGroup.isChecked()
        for c in self.curves:
            c.setSpectrumMode(b)
        self.enableAutoScale()
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
                    curves[i].free()
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
        
    def resizeEvent(self, ev):
        self.ctrlBtn.move(0, self.size().height() - self.ctrlBtn.size().height())
        self.autoBtn.move(self.ctrlBtn.width(), self.size().height() - self.autoBtn.size().height())
        
    def hoverMoveEvent(self, ev):
        self.mousePos = ev.pos()
        self.mouseScreenPos = ev.screenPos()
        
    def ctrlBtnClicked(self):
        self.ctrlMenu.popup(self.mouseScreenPos)

    def getLabel(self, key):
        pass
        
    def _checkScaleKey(self, key):
        if key not in self.scales:
            raise Exception("Scale '%s' not found. Scales are: %s" % (key, str(self.scales.keys())))
        
    def getScale(self, key):
        self._checkScaleKey(key)
        return self.scales[key]['item']
        
    def setLabel(self, key, text=None, units=None, unitPrefix=None, **args):
        self.getScale(key).setLabel(text=text, units=units, unitPrefix=unitPrefix, **args)
        
    def showLabel(self, key, show=True):
        self.getScale(key).showLabel(show)

    def setTitle(self, title=None, **args):
        if title is None:
            self.titleLabel.setVisible(False)
            self.layout.setRowFixedHeight(0, 0)
            self.titleLabel.setMaximumHeight(0)
        else:
            self.titleLabel.setMaximumHeight(30)
            self.layout.setRowFixedHeight(0, 30)
            self.titleLabel.setVisible(True)
            self.titleLabel.setText(title, **args)

    def showScale(self, key, show=True):
        s = self.getScale(key)
        p = self.scales[key]['pos']
        if show:
            s.show()
        else:
            s.hide()

    def _plotArray(self, arr, x=None):
        if arr.ndim != 1:
            raise Exception("Array must be 1D to plot (shape is %s)" % arr.shape)
        if x is None:
            x = np.arange(arr.shape[0])
        if x.ndim != 1:
            raise Exception("X array must be 1D to plot (shape is %s)" % x.shape)
        c = PlotCurveItem(arr, x=x)
        return c
            
        
        
    def _plotMetaArray(self, arr, x=None, autoLabel=True):
        inf = arr.infoCopy()
        if arr.ndim != 1:
            raise Exception('can only automatically plot 1 dimensional arrays.')
        ## create curve
        try:
            xv = arr.xvals(0)
            #print 'xvals:', xv
        except:
            if x is None:
                xv = arange(arr.shape[0])
            else:
                xv = x
        c = PlotCurveItem()
        c.setData(x=xv, y=arr.view(np.ndarray))
        
        if autoLabel:
            name = arr._info[0].get('name', None)
            units = arr._info[0].get('units', None)
            self.setLabel('bottom', text=name, units=units)
            
            name = arr._info[1].get('name', None)
            units = arr._info[1].get('units', None)
            self.setLabel('left', text=name, units=units)
            
        return c

    def saveSvgClicked(self):
        self.fileDialog = QtGui.QFileDialog()
        #if PlotItem.lastFileDir is not None:
            #self.fileDialog.setDirectory(PlotItem.lastFileDir)
        self.fileDialog.setFileMode(QtGui.QFileDialog.AnyFile)
        self.fileDialog.setAcceptMode(QtGui.QFileDialog.AcceptSave)
        if PlotItem.lastFileDir is not None:
            self.fileDialog.setDirectory(PlotItem.lastFileDir)
        self.fileDialog.show()
        QtCore.QObject.connect(self.fileDialog, QtCore.SIGNAL('fileSelected(const QString)'), self.writeSvg)
            
    #def svgFileSelected(self, fileName):
        ##PlotWidget.lastFileDir = os.path.split(fileName)[0]
        #self.writeSvg(str(fileName))

    def saveImgClicked(self):
        self.fileDialog = QtGui.QFileDialog()
        #if PlotItem.lastFileDir is not None:
            #self.fileDialog.setDirectory(PlotItem.lastFileDir)
        if PlotItem.lastFileDir is not None:
            self.fileDialog.setDirectory(PlotItem.lastFileDir)
        self.fileDialog.setFileMode(QtGui.QFileDialog.AnyFile)
        self.fileDialog.setAcceptMode(QtGui.QFileDialog.AcceptSave)
        self.fileDialog.show()
        QtCore.QObject.connect(self.fileDialog, QtCore.SIGNAL('fileSelected(const QString)'), self.writeImage)
            
    def saveCsvClicked(self):
        self.fileDialog = QtGui.QFileDialog()
        #if PlotItem.lastFileDir is not None:
            #self.fileDialog.setDirectory(PlotItem.lastFileDir)
        if PlotItem.lastFileDir is not None:
            self.fileDialog.setDirectory(PlotItem.lastFileDir)
        self.fileDialog.setFileMode(QtGui.QFileDialog.AnyFile)
        self.fileDialog.setAcceptMode(QtGui.QFileDialog.AcceptSave)
        self.fileDialog.show()
        QtCore.QObject.connect(self.fileDialog, QtCore.SIGNAL('fileSelected(const QString)'), self.writeCsv)
    #def imgFileSelected(self, fileName):
        ##PlotWidget.lastFileDir = os.path.split(fileName)[0]
        #self.writeImage(str(fileName))
      

class PlotWidgetManager(QtCore.QObject):
    """Used for managing communication between PlotWidgets"""
    def __init__(self):
        QtCore.QObject.__init__(self)
        self.widgets = weakref.WeakValueDictionary() # Don't keep PlotWidgets around just because they are listed here
    
    def addWidget(self, w, name):
        self.widgets[name] = w
        self.emit(QtCore.SIGNAL('widgetListChanged'), self.widgets.keys())
        
    def removeWidget(self, name):
        if name in self.widgets:
            del self.widgets[name]
            self.emit(QtCore.SIGNAL('widgetListChanged'), self.widgets.keys())
        
        
    def listWidgets(self):
        return self.widgets.keys()
        
    def getWidget(self, name):
        if name not in self.widgets:
            return None
        else:
            return self.widgets[name]
            
    def linkX(self, p1, p2):
        QtCore.QObject.connect(p1, QtCore.SIGNAL('xRangeChanged'), p2.linkXChanged)
        QtCore.QObject.connect(p2, QtCore.SIGNAL('xRangeChanged'), p1.linkXChanged)
        p1.linkXChanged(p2)
        #p2.setManualXScale()

    def unlinkX(self, p1, p2):
        QtCore.QObject.disconnect(p1, QtCore.SIGNAL('xRangeChanged'), p2.linkXChanged)
        QtCore.QObject.disconnect(p2, QtCore.SIGNAL('xRangeChanged'), p1.linkXChanged)
        
    def linkY(self, p1, p2):
        QtCore.QObject.connect(p1, QtCore.SIGNAL('yRangeChanged'), p2.linkYChanged)
        QtCore.QObject.connect(p2, QtCore.SIGNAL('yRangeChanged'), p1.linkYChanged)
        p1.linkYChanged(p2)
        #p2.setManualYScale()

    def unlinkY(self, p1, p2):
        QtCore.QObject.disconnect(p1, QtCore.SIGNAL('yRangeChanged'), p2.linkYChanged)
        QtCore.QObject.disconnect(p2, QtCore.SIGNAL('yRangeChanged'), p1.linkYChanged)
