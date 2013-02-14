from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.python2_3 import asUnicode
import numpy as np
from pyqtgraph.Point import Point
import pyqtgraph.debug as debug
import weakref
import pyqtgraph.functions as fn
import pyqtgraph as pg
from .GraphicsWidget import GraphicsWidget

__all__ = ['AxisItem']
class AxisItem(GraphicsWidget):
    """
    GraphicsItem showing a single plot axis with ticks, values, and label.
    Can be configured to fit on any side of a plot, and can automatically synchronize its displayed scale with ViewBox items.
    Ticks can be extended to draw a grid.
    If maxTickLength is negative, ticks point into the plot. 
    """
    
    def __init__(self, orientation, pen=None, linkView=None, parent=None, maxTickLength=-5, showValues=True):
        """
        ==============  ===============================================================
        **Arguments:**
        orientation     one of 'left', 'right', 'top', or 'bottom'
        maxTickLength   (px) maximum length of ticks to draw. Negative values draw
                        into the plot, positive values draw outward.
        linkView        (ViewBox) causes the range of values displayed in the axis
                        to be linked to the visible range of a ViewBox.
        showValues      (bool) Whether to display values adjacent to ticks 
        pen             (QPen) Pen used when drawing ticks.
        ==============  ===============================================================
        """
        
        GraphicsWidget.__init__(self, parent)
        self.label = QtGui.QGraphicsTextItem(self)
        self.showValues = showValues
        self.picture = None
        self.orientation = orientation
        if orientation not in ['left', 'right', 'top', 'bottom']:
            raise Exception("Orientation argument must be one of 'left', 'right', 'top', or 'bottom'.")
        if orientation in ['left', 'right']:
            #self.setMinimumWidth(25)
            #self.setSizePolicy(QtGui.QSizePolicy(
                #QtGui.QSizePolicy.Minimum,
                #QtGui.QSizePolicy.Expanding
            #))
            self.label.rotate(-90)
        #else:
            #self.setMinimumHeight(50)
            #self.setSizePolicy(QtGui.QSizePolicy(
                #QtGui.QSizePolicy.Expanding,
                #QtGui.QSizePolicy.Minimum
            #))
        #self.drawLabel = False
        
        self.labelText = ''
        self.labelUnits = ''
        self.labelUnitPrefix=''
        self.labelStyle = {}
        self.logMode = False
        self.tickFont = None
        
        self.textHeight = 18
        self.tickLength = maxTickLength
        self._tickLevels = None  ## used to override the automatic ticking system with explicit ticks
        self.scale = 1.0
        self.autoScale = True
        
        self.setRange(0, 1)
        
        self.setPen(pen)
        
        self._linkedView = None
        if linkView is not None:
            self.linkToView(linkView)
        
        self.showLabel(False)
        
        self.grid = False
        #self.setCacheMode(self.DeviceCoordinateCache)
        
    def close(self):
        self.scene().removeItem(self.label)
        self.label = None
        self.scene().removeItem(self)
        
    def setGrid(self, grid):
        """Set the alpha value for the grid, or False to disable."""
        self.grid = grid
        self.picture = None
        self.prepareGeometryChange()
        self.update()
        
    def setLogMode(self, log):
        """
        If *log* is True, then ticks are displayed on a logarithmic scale and values
        are adjusted accordingly. (This is usually accessed by changing the log mode 
        of a :func:`PlotItem <pyqtgraph.PlotItem.setLogMode>`)
        """
        self.logMode = log
        self.picture = None
        self.update()
        
    def setTickFont(self, font):
        self.tickFont = font
        self.picture = None
        self.prepareGeometryChange()
        ## Need to re-allocate space depending on font size?
        
        self.update()
        
    def resizeEvent(self, ev=None):
        #s = self.size()
        
        ## Set the position of the label
        nudge = 5
        br = self.label.boundingRect()
        p = QtCore.QPointF(0, 0)
        if self.orientation == 'left':
            p.setY(int(self.size().height()/2 + br.width()/2))
            p.setX(-nudge)
            #s.setWidth(10)
        elif self.orientation == 'right':
            #s.setWidth(10)
            p.setY(int(self.size().height()/2 + br.width()/2))
            p.setX(int(self.size().width()-br.height()+nudge))
        elif self.orientation == 'top':
            #s.setHeight(10)
            p.setY(-nudge)
            p.setX(int(self.size().width()/2. - br.width()/2.))
        elif self.orientation == 'bottom':
            p.setX(int(self.size().width()/2. - br.width()/2.))
            #s.setHeight(10)
            p.setY(int(self.size().height()-br.height()+nudge))
        #self.label.resize(s)
        self.label.setPos(p)
        self.picture = None
        
    def showLabel(self, show=True):
        """Show/hide the label text for this axis."""
        #self.drawLabel = show
        self.label.setVisible(show)
        if self.orientation in ['left', 'right']:
            self.setWidth()
        else:
            self.setHeight()
        if self.autoScale:
            self.setScale()
        
    def setLabel(self, text=None, units=None, unitPrefix=None, **args):
        """Set the text displayed adjacent to the axis.
        
        ============= =============================================================
        Arguments
        text          The text (excluding units) to display on the label for this
                      axis.
        units         The units for this axis. Units should generally be given
                      without any scaling prefix (eg, 'V' instead of 'mV'). The
                      scaling prefix will be automatically prepended based on the
                      range of data displayed.
        **args        All extra keyword arguments become CSS style options for 
                      the <span> tag which will surround the axis label and units.
        ============= =============================================================
        
        The final text generated for the label will look like::
        
            <span style="...options...">{text} (prefix{units})</span>
            
        Each extra keyword argument will become a CSS option in the above template. 
        For example, you can set the font size and color of the label::
        
            labelStyle = {'color': '#FFF', 'font-size': '14pt'}
            axis.setLabel('label text', units='V', **labelStyle)
        
        """
        if text is not None:
            self.labelText = text
            self.showLabel()
        if units is not None:
            self.labelUnits = units
            self.showLabel()
        if unitPrefix is not None:
            self.labelUnitPrefix = unitPrefix
        if len(args) > 0:
            self.labelStyle = args
        self.label.setHtml(self.labelString())
        self.resizeEvent()
        self.picture = None
        self.update()
            
    def labelString(self):
        if self.labelUnits == '':
            if self.scale == 1.0:
                units = ''
            else:
                units = asUnicode('(x%g)') % (1.0/self.scale)
        else:
            #print repr(self.labelUnitPrefix), repr(self.labelUnits)
            units = asUnicode('(%s%s)') % (self.labelUnitPrefix, self.labelUnits)
            
        s = asUnicode('%s %s') % (self.labelText, units)
        
        style = ';'.join(['%s: %s' % (k, self.labelStyle[k]) for k in self.labelStyle])
        
        return asUnicode("<span style='%s'>%s</span>") % (style, s)
        
    def setHeight(self, h=None):
        """Set the height of this axis reserved for ticks and tick labels.
        The height of the axis label is automatically added."""
        if h is None:
            h = self.textHeight + max(0, self.tickLength)
            if self.label.isVisible():
                h += self.textHeight
        self.setMaximumHeight(h)
        self.setMinimumHeight(h)
        self.picture = None
        
        
    def setWidth(self, w=None):
        """Set the width of this axis reserved for ticks and tick labels.
        The width of the axis label is automatically added."""
        if w is None:
            w = max(0, self.tickLength) + 40
            if self.label.isVisible():
                w += self.textHeight
        self.setMaximumWidth(w)
        self.setMinimumWidth(w)
        
    def pen(self):
        if self._pen is None:
            return fn.mkPen(pg.getConfigOption('foreground'))
        return pg.mkPen(self._pen)
        
    def setPen(self, pen):
        """
        Set the pen used for drawing text, axes, ticks, and grid lines.
        if pen == None, the default will be used (see :func:`setConfigOption 
        <pyqtgraph.setConfigOption>`)
        """
        self._pen = pen
        self.picture = None
        if pen is None:
            pen = pg.getConfigOption('foreground')
        self.labelStyle['color'] = '#' + pg.colorStr(pg.mkPen(pen).color())[:6]
        self.setLabel()
        self.update()
        
    def setScale(self, scale=None):
        """
        Set the value scaling for this axis. Values on the axis are multiplied
        by this scale factor before being displayed as text. By default,
        this scaling value is automatically determined based on the visible range
        and the axis units are updated to reflect the chosen scale factor.
        
        For example: If the axis spans values from -0.1 to 0.1 and has units set 
        to 'V' then a scale of 1000 would cause the axis to display values -100 to 100
        and the units would appear as 'mV'
        """
        if scale is None:
            #if self.drawLabel:  ## If there is a label, then we are free to rescale the values 
            if self.label.isVisible():
                #d = self.range[1] - self.range[0]
                #(scale, prefix) = fn.siScale(d / 2.)
                (scale, prefix) = fn.siScale(max(abs(self.range[0]), abs(self.range[1])))
                if self.labelUnits == '' and prefix in ['k', 'm']:  ## If we are not showing units, wait until 1e6 before scaling.
                    scale = 1.0
                    prefix = ''
                self.setLabel(unitPrefix=prefix)
            else:
                scale = 1.0
        else:
            self.setLabel(unitPrefix='')
            self.autoScale = False
            
        if scale != self.scale:
            self.scale = scale
            self.setLabel()
            self.picture = None
            self.update()
        
    def setRange(self, mn, mx):
        """Set the range of values displayed by the axis.
        Usually this is handled automatically by linking the axis to a ViewBox with :func:`linkToView <pyqtgraph.AxisItem.linkToView>`"""
        if any(np.isinf((mn, mx))) or any(np.isnan((mn, mx))):
            raise Exception("Not setting range to [%s, %s]" % (str(mn), str(mx)))
        self.range = [mn, mx]
        if self.autoScale:
            self.setScale()
        self.picture = None
        self.update()
        
    def linkedView(self):
        """Return the ViewBox this axis is linked to"""
        if self._linkedView is None:
            return None
        else:
            return self._linkedView()
        
    def linkToView(self, view):
        """Link this axis to a ViewBox, causing its displayed range to match the visible range of the view."""
        oldView = self.linkedView()
        self._linkedView = weakref.ref(view)
        if self.orientation in ['right', 'left']:
            if oldView is not None:
                oldView.sigYRangeChanged.disconnect(self.linkedViewChanged)
            view.sigYRangeChanged.connect(self.linkedViewChanged)
        else:
            if oldView is not None:
                oldView.sigXRangeChanged.disconnect(self.linkedViewChanged)
            view.sigXRangeChanged.connect(self.linkedViewChanged)
        
    def linkedViewChanged(self, view, newRange):
        if self.orientation in ['right', 'left'] and view.yInverted():
            self.setRange(*newRange[::-1])
        else:
            self.setRange(*newRange)
        
    def boundingRect(self):
        linkedView = self.linkedView()
        if linkedView is None or self.grid is False:
            rect = self.mapRectFromParent(self.geometry())
            ## extend rect if ticks go in negative direction
            ## also extend to account for text that flows past the edges
            if self.orientation == 'left':
                #rect.setRight(rect.right() - min(0,self.tickLength))
                #rect.setTop(rect.top() - 15)
                #rect.setBottom(rect.bottom() + 15)
                rect = rect.adjusted(0, -15, -min(0,self.tickLength), 15)
            elif self.orientation == 'right':
                #rect.setLeft(rect.left() + min(0,self.tickLength))
                rect = rect.adjusted(min(0,self.tickLength), -15, 0, 15)
            elif self.orientation == 'top':
                #rect.setBottom(rect.bottom() - min(0,self.tickLength))
                rect = rect.adjusted(-15, 0, 15, -min(0,self.tickLength))
            elif self.orientation == 'bottom':
                #rect.setTop(rect.top() + min(0,self.tickLength))
                rect = rect.adjusted(-15, min(0,self.tickLength), 15, 0)
            return rect
        else:
            return self.mapRectFromParent(self.geometry()) | linkedView.mapRectToItem(self, linkedView.boundingRect())
        
    def paint(self, p, opt, widget):
        if self.picture is None:
            self.picture = QtGui.QPicture()
            painter = QtGui.QPainter(self.picture)
            try:
                self.drawPicture(painter)
            finally:
                painter.end()
        #p.setRenderHint(p.Antialiasing, False)   ## Sometimes we get a segfault here ???
        #p.setRenderHint(p.TextAntialiasing, True)
        self.picture.play(p)
        

    def setTicks(self, ticks):
        """Explicitly determine which ticks to display.
        This overrides the behavior specified by tickSpacing(), tickValues(), and tickStrings()
        The format for *ticks* looks like::

            [
                [ (majorTickValue1, majorTickString1), (majorTickValue2, majorTickString2), ... ],
                [ (minorTickValue1, minorTickString1), (minorTickValue2, minorTickString2), ... ],
                ...
            ]
        
        If *ticks* is None, then the default tick system will be used instead.
        """
        self._tickLevels = ticks
        self.picture = None
        self.update()
    
    def tickSpacing(self, minVal, maxVal, size):
        """Return values describing the desired spacing and offset of ticks.
        
        This method is called whenever the axis needs to be redrawn and is a 
        good method to override in subclasses that require control over tick locations.
        
        The return value must be a list of three tuples::
        
            [
                (major tick spacing, offset),
                (minor tick spacing, offset),
                (sub-minor tick spacing, offset),
                ...
            ]
        """
        dif = abs(maxVal - minVal)
        if dif == 0:
            return []
        
        ## decide optimal minor tick spacing in pixels (this is just aesthetics)
        pixelSpacing = np.log(size+10) * 5
        optimalTickCount = max(2., size / pixelSpacing)
        
        ## optimal minor tick spacing 
        optimalSpacing = dif / optimalTickCount
        
        ## the largest power-of-10 spacing which is smaller than optimal
        p10unit = 10 ** np.floor(np.log10(optimalSpacing))
        
        ## Determine major/minor tick spacings which flank the optimal spacing.
        intervals = np.array([1., 2., 10., 20., 100.]) * p10unit
        minorIndex = 0
        while intervals[minorIndex+1] <= optimalSpacing:
            minorIndex += 1
            
        levels = [
            (intervals[minorIndex+2], 0),
            (intervals[minorIndex+1], 0),
            #(intervals[minorIndex], 0)    ## Pretty, but eats up CPU
        ]
        
        ## decide whether to include the last level of ticks
        minSpacing = min(size / 20., 30.)
        maxTickCount = size / minSpacing
        if dif / intervals[minorIndex] <= maxTickCount:
            levels.append((intervals[minorIndex], 0))
        return levels
        
        
        
        ##### This does not work -- switching between 2/5 confuses the automatic text-level-selection
        ### Determine major/minor tick spacings which flank the optimal spacing.
        #intervals = np.array([1., 2., 5., 10., 20., 50., 100.]) * p10unit
        #minorIndex = 0
        #while intervals[minorIndex+1] <= optimalSpacing:
            #minorIndex += 1
            
        ### make sure we never see 5 and 2 at the same time
        #intIndexes = [
            #[0,1,3],
            #[0,2,3],
            #[2,3,4],
            #[3,4,6],
            #[3,5,6],
        #][minorIndex]
        
        #return [
            #(intervals[intIndexes[2]], 0),
            #(intervals[intIndexes[1]], 0),
            #(intervals[intIndexes[0]], 0)
        #]
        
        

    def tickValues(self, minVal, maxVal, size):
        """
        Return the values and spacing of ticks to draw::
        
            [  
                (spacing, [major ticks]), 
                (spacing, [minor ticks]), 
                ... 
            ]
        
        By default, this method calls tickSpacing to determine the correct tick locations.
        This is a good method to override in subclasses.
        """
        minVal, maxVal = sorted((minVal, maxVal))
        
            
        ticks = []
        tickLevels = self.tickSpacing(minVal, maxVal, size)
        allValues = np.array([])
        for i in range(len(tickLevels)):
            spacing, offset = tickLevels[i]
            
            ## determine starting tick
            start = (np.ceil((minVal-offset) / spacing) * spacing) + offset
            
            ## determine number of ticks
            num = int((maxVal-start) / spacing) + 1
            values = np.arange(num) * spacing + start
            ## remove any ticks that were present in higher levels
            ## we assume here that if the difference between a tick value and a previously seen tick value
            ## is less than spacing/100, then they are 'equal' and we can ignore the new tick.
            values = list(filter(lambda x: all(np.abs(allValues-x) > spacing*0.01), values) )
            allValues = np.concatenate([allValues, values])
            ticks.append((spacing, values))
            
        if self.logMode:
            return self.logTickValues(minVal, maxVal, size, ticks)
            
        return ticks
    
    def logTickValues(self, minVal, maxVal, size, stdTicks):
        
        ## start with the tick spacing given by tickValues().
        ## Any level whose spacing is < 1 needs to be converted to log scale
        
        ticks = []
        for (spacing, t) in stdTicks:
            if spacing >= 1.0:
                ticks.append((spacing, t))
        
        if len(ticks) < 3:
            v1 = int(np.floor(minVal))
            v2 = int(np.ceil(maxVal))
            #major = list(range(v1+1, v2))
            
            minor = []
            for v in range(v1, v2):
                minor.extend(v + np.log10(np.arange(1, 10)))
            minor = [x for x in minor if x>minVal and x<maxVal]
            ticks.append((None, minor))
        return ticks

    def tickStrings(self, values, scale, spacing):
        """Return the strings that should be placed next to ticks. This method is called 
        when redrawing the axis and is a good method to override in subclasses.
        The method is called with a list of tick values, a scaling factor (see below), and the 
        spacing between ticks (this is required since, in some instances, there may be only 
        one tick and thus no other way to determine the tick spacing)
        
        The scale argument is used when the axis label is displaying units which may have an SI scaling prefix.
        When determining the text to display, use value*scale to correctly account for this prefix.
        For example, if the axis label's units are set to 'V', then a tick value of 0.001 might
        be accompanied by a scale value of 1000. This indicates that the label is displaying 'mV', and 
        thus the tick should display 0.001 * 1000 = 1.
        """
        if self.logMode:
            return self.logTickStrings(values, scale, spacing)
        
        places = max(0, np.ceil(-np.log10(spacing*scale)))
        strings = []
        for v in values:
            vs = v * scale
            if abs(vs) < .001 or abs(vs) >= 10000:
                vstr = "%g" % vs
            else:
                vstr = ("%%0.%df" % places) % vs
            strings.append(vstr)
        return strings
        
    def logTickStrings(self, values, scale, spacing):
        return ["%0.1g"%x for x in 10 ** np.array(values).astype(float)]
        
    def drawPicture(self, p):
        
        p.setRenderHint(p.Antialiasing, False)
        p.setRenderHint(p.TextAntialiasing, True)
        
        prof = debug.Profiler("AxisItem.paint", disabled=True)
        
        #bounds = self.boundingRect()
        bounds = self.mapRectFromParent(self.geometry())
        
        linkedView = self.linkedView()
        if linkedView is None or self.grid is False:
            tickBounds = bounds
        else:
            tickBounds = linkedView.mapRectToItem(self, linkedView.boundingRect())
        
        if self.orientation == 'left':
            span = (bounds.topRight(), bounds.bottomRight())
            tickStart = tickBounds.right()
            tickStop = bounds.right()
            tickDir = -1
            axis = 0
        elif self.orientation == 'right':
            span = (bounds.topLeft(), bounds.bottomLeft())
            tickStart = tickBounds.left()
            tickStop = bounds.left()
            tickDir = 1
            axis = 0
        elif self.orientation == 'top':
            span = (bounds.bottomLeft(), bounds.bottomRight())
            tickStart = tickBounds.bottom()
            tickStop = bounds.bottom()
            tickDir = -1
            axis = 1
        elif self.orientation == 'bottom':
            span = (bounds.topLeft(), bounds.topRight())
            tickStart = tickBounds.top()
            tickStop = bounds.top()
            tickDir = 1
            axis = 1
        #print tickStart, tickStop, span
        
        ## draw long line along axis
        p.setPen(self.pen())
        p.drawLine(*span)
        p.translate(0.5,0)  ## resolves some damn pixel ambiguity

        ## determine size of this item in pixels
        points = list(map(self.mapToDevice, span))
        if None in points:
            return
        lengthInPixels = Point(points[1] - points[0]).length()
        if lengthInPixels == 0:
            return

        if self._tickLevels is None:
            tickLevels = self.tickValues(self.range[0], self.range[1], lengthInPixels)
            tickStrings = None
        else:
            ## parse self.tickLevels into the formats returned by tickLevels() and tickStrings()
            tickLevels = []
            tickStrings = []
            for level in self._tickLevels:
                values = []
                strings = []
                tickLevels.append((None, values))
                tickStrings.append(strings)
                for val, strn in level:
                    values.append(val)
                    strings.append(strn)
        
        textLevel = 1  ## draw text at this scale level
        
        ## determine mapping between tick values and local coordinates
        dif = self.range[1] - self.range[0]
        if axis == 0:
            xScale = -bounds.height() / dif
            offset = self.range[0] * xScale - bounds.height()
        else:
            xScale = bounds.width() / dif
            offset = self.range[0] * xScale
            
        xRange = [x * xScale - offset for x in self.range]
        xMin = min(xRange)
        xMax = max(xRange)
        
        prof.mark('init')
            
        tickPositions = [] # remembers positions of previously drawn ticks
        
        ## draw ticks
        ## (to improve performance, we do not interleave line and text drawing, since this causes unnecessary pipeline switching)
        ## draw three different intervals, long ticks first
        
        for i in range(len(tickLevels)):
            tickPositions.append([])
            ticks = tickLevels[i][1]
        
            ## length of tick
            tickLength = self.tickLength / ((i*0.5)+1.0)
                
            lineAlpha = 255 / (i+1)
            if self.grid is not False:
                lineAlpha *= self.grid/255. * np.clip((0.05  * lengthInPixels / (len(ticks)+1)), 0., 1.)
            
            for v in ticks:
                ## determine actual position to draw this tick
                x = (v * xScale) - offset
                if x < xMin or x > xMax:  ## last check to make sure no out-of-bounds ticks are drawn
                    tickPositions[i].append(None)
                    continue
                tickPositions[i].append(x)
                
                p1 = [x, x]
                p2 = [x, x]
                p1[axis] = tickStart
                p2[axis] = tickStop
                if self.grid is False:
                    p2[axis] += tickLength*tickDir
                tickPen = self.pen()
                color = tickPen.color()
                color.setAlpha(lineAlpha)
                tickPen.setColor(color)
                p.setPen(tickPen)
                p.drawLine(Point(p1), Point(p2))
        prof.mark('draw ticks')

        ## Draw text until there is no more room (or no more text)
        if self.tickFont is not None:
            p.setFont(self.tickFont)
        
        textRects = []
        for i in range(len(tickLevels)):
            ## Get the list of strings to display for this level
            if tickStrings is None:
                spacing, values = tickLevels[i]
                strings = self.tickStrings(values, self.scale, spacing)
            else:
                strings = tickStrings[i]
                
            if len(strings) == 0:
                continue
            
            ## ignore strings belonging to ticks that were previously ignored
            for j in range(len(strings)):
                if tickPositions[i][j] is None:
                    strings[j] = None

            textRects.extend([p.boundingRect(QtCore.QRectF(0, 0, 100, 100), QtCore.Qt.AlignCenter, str(s)) for s in strings if s is not None])
            if i > 0:  ## always draw top level
                ## measure all text, make sure there's enough room
                if axis == 0:
                    textSize = np.sum([r.height() for r in textRects])
                else:
                    textSize = np.sum([r.width() for r in textRects])

                ## If the strings are too crowded, stop drawing text now
                textFillRatio = float(textSize) / lengthInPixels
                if textFillRatio > 0.7:
                    break
            #spacing, values = tickLevels[best]
            #strings = self.tickStrings(values, self.scale, spacing)
            for j in range(len(strings)):
                vstr = strings[j]
                if vstr is None: ## this tick was ignored because it is out of bounds
                    continue
                vstr = str(vstr)
                x = tickPositions[i][j]
                textRect = p.boundingRect(QtCore.QRectF(0, 0, 100, 100), QtCore.Qt.AlignCenter, vstr)
                height = textRect.height()
                self.textHeight = height
                if self.orientation == 'left':
                    textFlags = QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter
                    rect = QtCore.QRectF(tickStop-100, x-(height/2), 99-max(0,self.tickLength), height)
                elif self.orientation == 'right':
                    textFlags = QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter
                    rect = QtCore.QRectF(tickStop+max(0,self.tickLength)+1, x-(height/2), 100-max(0,self.tickLength), height)
                elif self.orientation == 'top':
                    textFlags = QtCore.Qt.AlignCenter|QtCore.Qt.AlignBottom
                    rect = QtCore.QRectF(x-100, tickStop-max(0,self.tickLength)-height, 200, height)
                elif self.orientation == 'bottom':
                    textFlags = QtCore.Qt.AlignCenter|QtCore.Qt.AlignTop
                    rect = QtCore.QRectF(x-100, tickStop+max(0,self.tickLength), 200, height)

                p.setPen(self.pen())
                p.drawText(rect, textFlags, vstr)
        prof.mark('draw text')
        prof.finish()
        
    def show(self):
        
        if self.orientation in ['left', 'right']:
            self.setWidth()
        else:
            self.setHeight()
        GraphicsWidget.show(self)
        
    def hide(self):
        if self.orientation in ['left', 'right']:
            self.setWidth(0)
        else:
            self.setHeight(0)
        GraphicsWidget.hide(self)

    def wheelEvent(self, ev):
        if self.linkedView() is None: 
            return
        if self.orientation in ['left', 'right']:
            self.linkedView().wheelEvent(ev, axis=1)
        else:
            self.linkedView().wheelEvent(ev, axis=0)
        ev.accept()
        
    def mouseDragEvent(self, event):
        if self.linkedView() is None: 
            return
        if self.orientation in ['left', 'right']:
            return self.linkedView().mouseDragEvent(event, axis=1)
        else:
            return self.linkedView().mouseDragEvent(event, axis=0)
        
    def mouseClickEvent(self, event):
        if self.linkedView() is None: 
            return
        return self.linkedView().mouseClickEvent(event)
