from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
from pyqtgraph.Point import Point
import pyqtgraph.debug as debug
import weakref
import pyqtgraph.functions as fn
from GraphicsWidget import GraphicsWidget

__all__ = ['AxisItem']
class AxisItem(GraphicsWidget):
    def __init__(self, orientation, pen=None, linkView=None, parent=None, maxTickLength=-5, showValues=True):
        """
        GraphicsItem showing a single plot axis with ticks, values, and label.
        Can be configured to fit on any side of a plot, and can automatically synchronize its displayed scale with ViewBox items.
        Ticks can be extended to make a grid.
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
        #self.drawLabel = show
        self.label.setVisible(show)
        if self.orientation in ['left', 'right']:
            self.setWidth()
        else:
            self.setHeight()
        if self.autoScale:
            self.setScale()
        
    def setLabel(self, text=None, units=None, unitPrefix=None, **args):
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
        For example:
            If the axis spans values from -0.1 to 0.1 and has units set to 'V'
            then a scale of 1000 would cause the axis to display values -100 to 100
            and the units would appear as 'mV'
        If scale is None, then it will be determined automatically based on the current 
        range displayed by the axis.
        """
        if scale is None:
            #if self.drawLabel:  ## If there is a label, then we are free to rescale the values 
            if self.label.isVisible():
                d = self.range[1] - self.range[0]
                #pl = 1-int(log10(d))
                #scale = 10 ** pl
                (scale, prefix) = fn.siScale(d / 2.)
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
        
        
    def drawPicture(self, p):
        
        prof = debug.Profiler("AxisItem.paint", disabled=True)
        p.setPen(self.pen)
        
        #bounds = self.boundingRect()
        bounds = self.mapRectFromParent(self.geometry())
        
        linkedView = self.linkedView()
        if linkedView is None or self.grid is False:
            tbounds = bounds
        else:
            tbounds = linkedView.mapRectToItem(self, linkedView.boundingRect())
        
        if self.orientation == 'left':
            span = (bounds.topRight(), bounds.bottomRight())
            tickStart = tbounds.right()
            tickStop = bounds.right()
            tickDir = -1
            axis = 0
        elif self.orientation == 'right':
            span = (bounds.topLeft(), bounds.bottomLeft())
            tickStart = tbounds.left()
            tickStop = bounds.left()
            tickDir = 1
            axis = 0
        elif self.orientation == 'top':
            span = (bounds.bottomLeft(), bounds.bottomRight())
            tickStart = tbounds.bottom()
            tickStop = bounds.bottom()
            tickDir = -1
            axis = 1
        elif self.orientation == 'bottom':
            span = (bounds.topLeft(), bounds.topRight())
            tickStart = tbounds.top()
            tickStop = bounds.top()
            tickDir = 1
            axis = 1

        ## draw long line along axis
        p.drawLine(*span)

        ## determine size of this item in pixels
        points = map(self.mapToDevice, span)
        lengthInPixels = Point(points[1] - points[0]).length()

        ## decide optimal tick spacing in pixels
        pixelSpacing = np.log(lengthInPixels+10) * 2
        optimalTickCount = lengthInPixels / pixelSpacing

        ## Determine optimal tick spacing
        #intervals = [1., 2., 5., 10., 20., 50.]
        #intervals = [1., 2.5, 5., 10., 25., 50.]
        intervals = np.array([0.1, 0.2, 1., 2., 10., 20., 100., 200.])
        dif = abs(self.range[1] - self.range[0])
        if dif == 0.0:
            return
        pw = 10 ** (np.floor(np.log10(dif))-1)
        scaledIntervals = intervals * pw
        scaledTickCounts = dif / scaledIntervals 
        try:
            i1 = np.argwhere(scaledTickCounts < optimalTickCount)[0,0]
        except:
            print "AxisItem can't determine tick spacing:"
            print "scaledTickCounts", scaledTickCounts
            print "optimalTickCount", optimalTickCount
            print "dif", dif
            print "scaledIntervals", scaledIntervals
            print "intervals", intervals
            print "pw", pw
            print "pixelSpacing", pixelSpacing
            i1 = 1
        
        distBetweenIntervals = (optimalTickCount-scaledTickCounts[i1]) / (scaledTickCounts[i1-1]-scaledTickCounts[i1])
        
        #print optimalTickCount, i1, scaledIntervals, distBetweenIntervals
        
        #for i in range(len(intervals)):
            #i1 = i
            #if dif / (pw*intervals[i]) < 10:
                #break
        
        textLevel = 0  ## draw text at this scale level
        
        #print "range: %s   dif: %f   power: %f  interval: %f   spacing: %f" % (str(self.range), dif, pw, intervals[i1], sp)
        
        #print "  start at %f,  %d ticks" % (start, num)
        
        
        if axis == 0:
            xs = -bounds.height() / dif
        else:
            xs = bounds.width() / dif
        
        prof.mark('init')
            
        tickPositions = set() # remembers positions of previously drawn ticks
        ## draw ticks and generate list of texts to draw
        ## (to improve performance, we do not interleave line and text drawing, since this causes unnecessary pipeline switching)
        ## draw three different intervals, long ticks first
        texts = []
        for i in [2,1,0]:
            if i1+i >= len(intervals) or i1+i < 0:
                print "AxisItem.paint error: i1=%d, i=%d, len(intervals)=%d" % (i1, i, len(intervals))
                continue
            ## spacing for this interval
            
            sp = pw*intervals[i1+i]
            
            ## determine starting tick
            start = np.ceil(self.range[0] / sp) * sp
            
            ## determine number of ticks
            num = int(dif / sp) + 1
            
            ## last tick value
            last = start + sp * num
            
            ## Number of decimal places to print
            maxVal = max(abs(start), abs(last))
            places = max(0, 1-int(np.log10(sp*self.scale)))
        
            ## length of tick
            #h = np.clip((self.tickLength*3 / num) - 1., min(0, self.tickLength), max(0, self.tickLength))
            if i == 0:
                h = self.tickLength * distBetweenIntervals / 2.
            else:
                h = self.tickLength*i/2.
                
            ## alpha
            if i == 0:
                #a = min(255, (765. / num) - 1.)
                a = 255 * distBetweenIntervals
            else:
                a = 255
                
            lineAlpha = a
            textAlpha = a
                
            if self.grid is not False:
                print self.grid
                lineAlpha = int(lineAlpha * self.grid / 255.)
            
            if axis == 0:
                offset = self.range[0] * xs - bounds.height()
            else:
                offset = self.range[0] * xs
            
            for j in range(num):
                v = start + sp * j
                x = (v * xs) - offset
                p1 = [0, 0]
                p2 = [0, 0]
                p1[axis] = tickStart
                p2[axis] = tickStop
                if self.grid is False:
                    p2[axis] += h*tickDir
                p1[1-axis] = p2[1-axis] = x
                
                if p1[1-axis] > [bounds.width(), bounds.height()][1-axis]:
                    continue
                if p1[1-axis] < 0:
                    continue
                p.setPen(QtGui.QPen(QtGui.QColor(150, 150, 150, lineAlpha)))
                # draw tick only if there is none
                tickPos = p1[1-axis]
                if tickPos not in tickPositions:
                    p.drawLine(Point(p1), Point(p2))
                    tickPositions.add(tickPos)
                    if i >= textLevel:
                        if abs(v) < .001 or abs(v) >= 10000:
                            vstr = "%g" % (v * self.scale)
                        else:
                            vstr = ("%%0.%df" % places) % (v * self.scale)
                            
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
                        
                        #p.setPen(QtGui.QPen(QtGui.QColor(150, 150, 150, a)))
                        #p.drawText(rect, textFlags, vstr)
                        texts.append((rect, textFlags, vstr, textAlpha))
                    
        prof.mark('draw ticks')
        for args in texts:
            p.setPen(QtGui.QPen(QtGui.QColor(150, 150, 150, args[3])))
            p.drawText(*args[:3])
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
