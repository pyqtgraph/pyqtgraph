from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
from pyqtgraph.Point import Point
import pyqtgraph.debug as debug
import weakref
import pyqtgraph.functions as fn
from GraphicsWidget import GraphicsWidget

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
        self.labelStyle = {'color': '#CCC'}
        self.logMode = False
        
        self.textHeight = 18
        self.tickLength = maxTickLength
        self.scale = 1.0
        self.autoScale = True
            
        self.setRange(0, 1)
        
        if pen is None:
            pen = QtGui.QPen(QtGui.QColor(100, 100, 100))
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
        """Set the text displayed adjacent to the axis."""
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
                units = u'(x%g)' % (1.0/self.scale)
        else:
            #print repr(self.labelUnitPrefix), repr(self.labelUnits)
            units = u'(%s%s)' % (self.labelUnitPrefix, self.labelUnits)
            
        s = u'%s %s' % (self.labelText, units)
        
        style = ';'.join(['%s: "%s"' % (k, self.labelStyle[k]) for k in self.labelStyle])
        
        return u"<span style='%s'>%s</span>" % (style, s)
        
    def setHeight(self, h=None):
        if h is None:
            h = self.textHeight + max(0, self.tickLength)
            if self.label.isVisible():
                h += self.textHeight
        self.setMaximumHeight(h)
        self.setMinimumHeight(h)
        self.picture = None
        
        
    def setWidth(self, w=None):
        if w is None:
            w = max(0, self.tickLength) + 40
            if self.label.isVisible():
                w += self.textHeight
        self.setMaximumWidth(w)
        self.setMinimumWidth(w)
        
    def setPen(self, pen):
        self.pen = pen
        self.picture = None
        self.update()
        
    def setScale(self, scale=None):
        """
        Set the value scaling for this axis. 
        The scaling value 1) multiplies the values displayed along the axis
        and 2) changes the way units are displayed in the label. 
        
        For example: If the axis spans values from -0.1 to 0.1 and has units set 
        to 'V' then a scale of 1000 would cause the axis to display values -100 to 100
        and the units would appear as 'mV'
        
        If scale is None, then it will be determined automatically based on the current 
        range displayed by the axis.
        """
        if scale is None:
            #if self.drawLabel:  ## If there is a label, then we are free to rescale the values 
            if self.label.isVisible():
                d = self.range[1] - self.range[0]
                #(scale, prefix) = fn.siScale(d / 2.)
                (scale, prefix) = fn.siScale(max(abs(self.range[0]), abs(self.range[1])))
                if self.labelUnits == '' and prefix in ['k', 'm']:  ## If we are not showing units, wait until 1e6 before scaling.
                    scale = 1.0
                    prefix = ''
                self.setLabel(unitPrefix=prefix)
            else:
                scale = 1.0
        
        
        if scale != self.scale:
            self.scale = scale
            self.setLabel()
            self.picture = None
            self.update()
        
    def setRange(self, mn, mx):
        """Set the range of values displayed by the axis"""
        if mn in [np.nan, np.inf, -np.inf] or mx in [np.nan, np.inf, -np.inf]:
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
        self.setRange(*newRange)
        
    def boundingRect(self):
        linkedView = self.linkedView()
        if linkedView is None or self.grid is False:
            rect = self.mapRectFromParent(self.geometry())
            ## extend rect if ticks go in negative direction
            if self.orientation == 'left':
                rect.setRight(rect.right() - min(0,self.tickLength))
            elif self.orientation == 'right':
                rect.setLeft(rect.left() + min(0,self.tickLength))
            elif self.orientation == 'top':
                rect.setBottom(rect.bottom() - min(0,self.tickLength))
            elif self.orientation == 'bottom':
                rect.setTop(rect.top() + min(0,self.tickLength))
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
        self.picture.play(p)
        


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
        optimalTickCount = size / pixelSpacing
        if optimalTickCount < 1:
            optimalTickCount = 1
        
        ## optimal minor tick spacing 
        optimalSpacing = dif / optimalTickCount
        
        ## the largest power-of-10 spacing which is smaller than optimal
        p10unit = 10 ** np.floor(np.log10(optimalSpacing))
        
        ## Determine major/minor tick spacings which flank the optimal spacing.
        intervals = np.array([1., 2., 10., 20., 100.]) * p10unit
        minorIndex = 0
        while intervals[minorIndex+1] <= optimalSpacing:
            minorIndex += 1
            
        return [
            (intervals[minorIndex+2], 0),
            (intervals[minorIndex+1], 0),
            (intervals[minorIndex], 0)
        ]
        

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
        if self.logMode:
            return self.logTickValues(minVal, maxVal, size)
            
        ticks = []
        tickLevels = self.tickSpacing(minVal, maxVal, size)
        for i in range(len(tickLevels)):
            spacing, offset = tickLevels[i]
            
            ## determine starting tick
            start = (np.ceil((minVal-offset) / spacing) * spacing) + offset
            
            ## determine number of ticks
            num = int((maxVal-start) / spacing) + 1
            ticks.append((spacing, np.arange(num) * spacing + start))
        return ticks
    
    def logTickValues(self, minVal, maxVal, size):
        v1 = int(np.floor(minVal))
        v2 = int(np.ceil(maxVal))
        major = range(v1+1, v2)
        
        minor = []
        for v in range(v1, v2):
            minor.extend(v + np.log10(np.arange(1, 10)))
        minor = filter(lambda x: x>minVal and x<maxVal, minor)
        return [(1.0, major), (None, minor)]

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
        p.setPen(self.pen)
        
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
        p.drawLine(*span)
        p.translate(0.5,0)  ## resolves some damn pixel ambiguity

        ## determine size of this item in pixels
        points = map(self.mapToDevice, span)
        lengthInPixels = Point(points[1] - points[0]).length()
        if lengthInPixels == 0:
            return


        tickLevels = self.tickValues(self.range[0], self.range[1], lengthInPixels)
        
        textLevel = 1  ## draw text at this scale level
        
        ## determine mapping between tick values and local coordinates
        dif = self.range[1] - self.range[0]
        if axis == 0:
            xScale = -bounds.height() / dif
            offset = self.range[0] * xScale - bounds.height()
        else:
            xScale = bounds.width() / dif
            offset = self.range[0] * xScale
        
        prof.mark('init')
            
        tickPositions = [] # remembers positions of previously drawn ticks
        
        ## draw ticks
        ## (to improve performance, we do not interleave line and text drawing, since this causes unnecessary pipeline switching)
        ## draw three different intervals, long ticks first
        for i in range(len(tickLevels)):
            tickPositions.append([])
            ticks = tickLevels[i][1]
        
            ## length of tick
            tickLength = self.tickLength / ((i*1.0)+1.0)
                
            lineAlpha = 255 / (i+1)
            if self.grid is not False:
                lineAlpha = self.grid
            
            for v in ticks:
                x = (v * xScale) - offset
                p1 = [x, x]
                p2 = [x, x]
                p1[axis] = tickStart
                p2[axis] = tickStop
                if self.grid is False:
                    p2[axis] += tickLength*tickDir
                p.setPen(QtGui.QPen(QtGui.QColor(150, 150, 150, lineAlpha)))
                p.drawLine(Point(p1), Point(p2))
                tickPositions[i].append(x)
        prof.mark('draw ticks')
        
        ## determine level to draw text
        best = 0
        for i in range(len(tickLevels)):
            ## take a small sample of strings and measure their rendered text
            spacing, values = tickLevels[i]
            strings = self.tickStrings(values, self.scale, spacing)
            if len(strings) == 0:
                continue
            textRects = [p.boundingRect(QtCore.QRectF(0, 0, 100, 100), QtCore.Qt.AlignCenter, s) for s in strings]
            if axis == 0:
                textSize = np.max([r.height() for r in textRects])
            else:
                textSize = np.max([r.width() for r in textRects])
                
            ## If these strings are not too crowded, then this level is ok
            textFillRatio = float(textSize * len(values)) / lengthInPixels
            if textFillRatio < 0.7:
                best = i
                continue
        prof.mark('measure text')
            
        spacing, values = tickLevels[best]
        strings = self.tickStrings(values, self.scale, spacing)
        for j in range(len(strings)):
            vstr = strings[j]
            x = tickPositions[best][j]
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
            
            p.setPen(QtGui.QPen(QtGui.QColor(150, 150, 150)))
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
