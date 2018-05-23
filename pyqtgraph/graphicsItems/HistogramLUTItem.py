"""
GraphicsWidget displaying an image histogram along with gradient editor. Can be used to adjust the appearance of images.
"""


from ..Qt import QtGui, QtCore
from .. import functions as fn
from .GraphicsWidget import GraphicsWidget
from .ViewBox import *
from .GradientEditorItem import *
from .LinearRegionItem import *
from .PlotDataItem import *
from .AxisItem import *
from .GridItem import *
from ..Point import Point
from .. import functions as fn
import numpy as np
from .. import debug as debug

import weakref

__all__ = ['HistogramLUTItem']


class HistogramLUTItem(GraphicsWidget):
    """
    This is a graphicsWidget which provides controls for adjusting the display of an image.
    
    Includes:

    - Image histogram 
    - Movable region over histogram to select black/white levels
    - Gradient editor to define color lookup table for single-channel images
    
    Parameters
    ----------
    image : ImageItem or None
        If *image* is provided, then the control will be automatically linked to
        the image and changes to the control will be immediately reflected in
        the image's appearance.
    fillHistogram : bool
        By default, the histogram is rendered with a fill.
        For performance, set *fillHistogram* = False.    
    rgbHistogram : bool
        Sets whether the histogram is computed once over all channels of the
        image, or once per channel.
    levelMode : 'mono' or 'rgba'
        If 'mono', then only a single set of black/whilte level lines is drawn,
        and the levels apply to all channels in the image. If 'rgba', then one
        set of levels is drawn for each channel.
    """
    
    sigLookupTableChanged = QtCore.Signal(object)
    sigLevelsChanged = QtCore.Signal(object)
    sigLevelChangeFinished = QtCore.Signal(object)
    sigLogModeChanged = QtCore.Signal(object)
    sigAutoLevelsChanged = QtCore.Signal(object)
    
    def __init__(self, image=None, fillHistogram=True, rgbHistogram=False, levelMode='mono'):
        GraphicsWidget.__init__(self)
        self.lut = None
        self.range = None
        self.imageItem = lambda: None  # fake a dead weakref
        self.levelMode = levelMode
        self.rgbHistogram = rgbHistogram
        self.blockAutoLevelsSignal = False
        
        self.layout = QtGui.QGraphicsGridLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(1,1,1,1)
        self.layout.setSpacing(0)
        self.vb = ViewBox(parent=self)
        self.vb.setMaximumWidth(152)
        self.vb.setMinimumWidth(45)
        self.vb.setMouseEnabled(x=False, y=True)
        self.gradient = GradientEditorItem(histogram=self)
        self.gradient.setOrientation('right')
        self.gradient.loadPreset('grey')
        self.regions = [
            LinearRegionItem([0, 1], 'horizontal', swapMode='block'),
            LinearRegionItem([0, 1], 'horizontal', swapMode='block', pen='r',
                             brush=fn.mkBrush((255, 50, 50, 50)), span=(0., 1/3.)),
            LinearRegionItem([0, 1], 'horizontal', swapMode='block', pen='g',
                             brush=fn.mkBrush((50, 255, 50, 50)), span=(1/3., 2/3.)),
            LinearRegionItem([0, 1], 'horizontal', swapMode='block', pen='b',
                             brush=fn.mkBrush((50, 50, 255, 80)), span=(2/3., 1.)),
            LinearRegionItem([0, 1], 'horizontal', swapMode='block', pen='w',
                             brush=fn.mkBrush((255, 255, 255, 50)), span=(2/3., 1.))]
        for region in self.regions:
            region.setZValue(1000)
            self.vb.addItem(region)
            region.lines[0].addMarker('<|', 0.5)
            region.lines[1].addMarker('|>', 0.5)
            region.sigRegionChanged.connect(self.regionChanging)
            region.sigRegionChangeFinished.connect(self.regionChanged)
            
        self.region = self.regions[0]  # for backward compatibility.
        
        self.axis = AxisItem('left', linkView=self.vb, maxTickLength=-10, parent=self)
        self.layout.addItem(self.axis, 0, 0)
        self.layout.addItem(self.vb, 0, 1)
        self.layout.addItem(self.gradient, 0, 2)
        self.gradient.setFlag(self.gradient.ItemStacksBehindParent)
        self.vb.setFlag(self.gradient.ItemStacksBehindParent)
        
        self.gradient.sigGradientChanged.connect(self.gradientChanged)
        self.gradient.sigLogModeChanged.connect(self.logModeChanged)
        self.gradient.sigAutoLevelsChanged.connect(self.autoLevelsChanged)
        self.vb.sigRangeChanged.connect(self.viewRangeChanged)
        self.vb.sigLogChanged.connect(self.viewLogChanged)
        add = QtGui.QPainter.CompositionMode_Plus
        self.plots = [
            PlotCurveItem(pen=(200, 200, 200, 100)),  # mono
            PlotCurveItem(pen=(255, 0, 0, 100), compositionMode=add),  # r
            PlotCurveItem(pen=(0, 255, 0, 100), compositionMode=add),  # g
            PlotCurveItem(pen=(0, 0, 255, 100), compositionMode=add),  # b
            PlotCurveItem(pen=(200, 200, 200, 100), compositionMode=add),  # a
            ]
        
        self.plot = self.plots[0]  # for backward compatibility.
        for plot in self.plots:
            plot.rotate(90)
            self.vb.addItem(plot)
        
        self.fillHistogram(fillHistogram)
        self._showRegions()
            
        self.vb.addItem(self.plot)
        self.autoHistogramRange()
        
        if image is not None:
            self.setImageItem(image)
        
    def fillHistogram(self, fill=True, level=0.0, color=(100, 100, 200)):
        colors = [color, (255, 0, 0, 50), (0, 255, 0, 50), (0, 0, 255, 50), (255, 255, 255, 50)]
        for i,plot in enumerate(self.plots):
            if fill:
                plot.setFillLevel(level)
                plot.setBrush(colors[i])
            else:
                plot.setFillLevel(None)
        
    def paint(self, p, *args):
        if self.levelMode != 'mono':
            return
        
        pen = self.region.lines[0].pen
        rgn = self.region.dataBounds(self.region.orientation)
        p1 = self.vb.mapFromViewToItem(self, Point(self.vb.viewRect().center().x(), rgn[0]))
        p2 = self.vb.mapFromViewToItem(self, Point(self.vb.viewRect().center().x(), rgn[1]))
        gradRect = self.gradient.mapRectToParent(self.gradient.gradRect.rect())
        for pen in [fn.mkPen((0, 0, 0, 100), width=3), pen]:
            p.setPen(pen)
            p.drawLine(p1 + Point(0, 5), gradRect.bottomLeft())
            p.drawLine(p2 - Point(0, 5), gradRect.topLeft())
            p.drawLine(gradRect.topLeft(), gradRect.topRight())
            p.drawLine(gradRect.bottomLeft(), gradRect.bottomRight())
        
    def setHistogramRange(self, mn, mx, padding=0.1):
        """Set the Y range on the histogram plot. This disables auto-scaling."""
        self.vb.enableAutoRange(self.vb.YAxis, False)
        self.vb.setYRange(mn, mx, padding)
        
    def autoHistogramRange(self):
        """Enable auto-scaling on the histogram plot."""
        self.vb.enableAutoRange(self.vb.XYAxes)

    def setImageItem(self, img):
        """Set an ImageItem to have its levels and LUT automatically controlled
        by this HistogramLUTItem.
        """
        self.imageItem = weakref.ref(img)
        img.sigImageChanged.connect(self.imageChanged)
        img.setLookupTable(self.getLookupTable)  ## send function pointer, not the result
        self.regionChanged()
        self.imageChanged(autoLevel=True)
        
    def viewRangeChanged(self):
        self.update()

    def viewLogChanged(self):
        x = self.vb.xLog()
        y = self.vb.yLog()
        for plot in self.plots:
            plot.setLogMode(y,x)
        self.region.setLogMode(x,y)
        self.axis.setLogMode(y)
        #are = self.vb.autoRangeEnabled()
        #self.vb.enableAutoRange()
        #self.vb.enableAutoRange(x=are[0],y=are[1])
        self.updatePlots()
    
    def gradientChanged(self):
        if self.imageItem() is not None:
            if self.gradient.isLookupTrivial():
                self.imageItem().setLookupTable(None) #lambda x: x.astype(np.uint8))
            else:
                self.imageItem().setLookupTable(self.getLookupTable)  ## send function pointer, not the result
            
        self.lut = None
        self.sigLookupTableChanged.emit(self)

    def getLookupTable(self, img=None, n=None, alpha=None):
        """Return a lookup table from the color gradient defined by this 
        HistogramLUTItem.
        """
        if self.levelMode is not 'mono':
            return None
        if n is None:
            if img.dtype == np.uint8:
                n = 256
            else:
                n = 512
        if self.lut is None:
            self.lut = self.gradient.getLookupTable(n, alpha=alpha)
        return self.lut

    def regionChanged(self):
        if self.imageItem() is not None:
            self.imageItem().setLevels(self.region.getRegion())
        self.sigLevelChangeFinished.emit(self)
        if not self.blockAutoLevelsSignal:
            self.gradient.setAutoLevels(False)

    def regionChanging(self):
        if self.imageItem() is not None:
            self.imageItem().setLevels(self.region.getRegion())
        self.sigLevelsChanged.emit(self)
        self.update()

    def imageChanged(self, autoLevel=False, autoRange=False):
        self.updatePlots()
        if autoLevel or self.autoLevelsEnabled():
            self.updateAutoLevels()

    def getLevels(self):
        """Return the min and max levels.
        
        For rgba mode, this returns a list of the levels for each channel.
        """
        if self.levelMode == 'mono':
            return self.region.getRegion()
        else:
            nch = self.imageItem().channels()
            if nch is None:
                nch = 3
            return [r.getRegion() for r in self.regions[1:nch+1]]
        
    def setLevels(self, min=None, max=None, rgba=None):
        """Set the min/max (bright and dark) levels.
        
        Arguments may be *min* and *max* for single-channel data, or 
        *rgba* = [(rmin, rmax), ...] for multi-channel data.
        """
        if self.levelMode == 'mono':
            if min is None:
                min, max = rgba[0]
            assert None not in (min, max)
            self.region.setRegion((min, max))
            self.gradient.setAutoLevels(False)
        else:
            if rgba is None:
                raise TypeError("Must specify rgba argument when levelMode != 'mono'.")
            for i, levels in enumerate(rgba):
                self.regions[i+1].setRegion(levels)
        
    def setLevelMode(self, mode):
        """ Set the method of controlling the image levels offered to the user. 
        Options are 'mono' or 'rgba'.
        """
        assert mode in ('mono', 'rgba')
        
        if mode == self.levelMode:
            return
        
        oldLevels = self.getLevels()
        self.levelMode = mode
        self._showRegions()
        
        # do our best to preserve old levels
        if mode == 'mono':
            levels = np.array(oldLevels).mean(axis=0)
            self.setLevels(*levels)
        else:
            levels = [oldLevels] * 4
            self.setLevels(rgba=levels)
            
        # force this because calling self.setLevels might not set the imageItem
        # levels if there was no change to the region item
        self.imageItem().setLevels(self.getLevels())
        
        self.imageChanged()
        self.update()

    def _showRegions(self):
        for i in range(len(self.regions)):
            self.regions[i].setVisible(False)
            
        if self.levelMode == 'rgba':
            imax = 4
            if self.imageItem() is not None:
                # Only show rgb channels if connected image lacks alpha.
                nch = self.imageItem().channels()
                if nch is None:
                    nch = 3
            xdif = 1.0 / nch
            for i in range(1, nch+1):
                self.regions[i].setVisible(True)
                self.regions[i].setSpan((i-1) * xdif, i * xdif)
            self.gradient.hide()
        elif self.levelMode == 'mono':
            self.regions[0].setVisible(True)
            self.gradient.show()
        else:
            raise ValueError("Unknown level mode %r" %  self.levelMode) 
    
    def saveState(self):
        return {
            'gradient': self.gradient.saveState(),
            'levels': self.getLevels(),
            'mode': self.levelMode,
        }
    
    def restoreState(self, state):
        self.setLevelMode(state['mode'])
        self.gradient.restoreState(state['gradient'])
        self.setLevels(*state['levels'])

    def autoLevelsChanged(self):
        self.updateAutoLevels()
        self.sigAutoLevelsChanged.emit(self)

    def autoLevelsEnabled(self):
        return self.gradient.autoLevelsEnabled()

    def setAutoLevels(self, value):
        self.gradient.setAutoLevels(value)
        if value:
            self.updateAutoLevels()

    def clearPlots(self):
        for plot in self.plots:
            plot.clear()

    def updatePlots(self):
        if self.imageItem() is None:
            self.clearPlots()
            return

        if self.levelMode == 'mono':
            for plt in self.plots[1:]:
                plt.setVisible(False)
            self.plots[0].setVisible(True)
            # plot one histogram for all image data
            profiler = debug.Profiler()
            h = self.imageItem().getHistogram()
            profiler('get histogram')
            if h[0] is None:
                self.clearPlots()
                return
            self.plot.setData(*h)
            profiler('set plot')
        else:
            # plot one histogram for each channel
            self.plots[0].setVisible(False)
            ch = self.imageItem().getHistogram(perChannel=True)
            if ch[0] is None:
                self.clearPlots()
                return
            for i in range(1, 5):
                if len(ch) >= i:
                    h = ch[i-1]
                    self.plots[i].setVisible(True)
                    self.plots[i].setData(*h)
                else:
                    # hide channels not present in image data
                    self.plots[i].setVisible(False)
            # make sure we are displaying the correct number of channels
            self._showRegions()

    def quickMinMax(self, data, axis=None):
        """
        Estimate the min/max values of *data* by subsampling.
        Returns [(min, max), ...] with one item per channel
        """
        while data.size > 1e6:
            ax = np.argmax(data.shape)
            sl = [slice(None)] * data.ndim
            sl[ax] = slice(None, None, 2)
            data = data[sl]

        if self.logModeEnabled():
            data = data[data > 0]

        if axis is None:
            return [(float(np.nanmin(data)), float(np.nanmax(data)))]
        else:
            return [(float(np.nanmin(data.take(i, axis=axis))),
                     float(np.nanmax(data.take(i, axis=axis)))) for i in range(data.shape[-1])]

    def autoLevels(self, axis=None):
        levels = self.quickMinMax(self.imageItem().image)[0]
        self.setLevels(*levels, rgba=self.quickMinMax(self.imageItem().image, axis))

    def autoRange(self):
        self.vb.autoRange()

    def updateAutoLevels(self):
        image = self.imageItem().image
        if image is None:
            return
        profiler = debug.Profiler()
        self.blockAutoLevelsSignal = True
        self.region.setRegion(self.quickMinMax(image)[0])
        self.blockAutoLevelsSignal = False
        profiler('set region')

    def logModeChanged(self):
        self.imageItem().setLog(self.gradient.logModeEnabled())
        self.sigLogModeChanged.emit(self)

    def logModeEnabled(self):
        return self.gradient.logModeEnabled()

    def setLogMode(self, value):
        self.imageItem().setLog(value)
        self.gradient.setLogMode(value)
        self.updatePlots()
