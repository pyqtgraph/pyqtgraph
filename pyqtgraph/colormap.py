import numpy as np
import scipy.interpolate
from pyqtgraph.Qt import QtGui, QtCore

class ColorMap(object):
    
    ## color interpolation modes
    RGB = 1
    HSV_POS = 2
    HSV_NEG = 3
    
    ## boundary modes
    CLIP = 1
    REPEAT = 2
    MIRROR = 3
    
    ## return types
    BYTE = 1
    FLOAT = 2
    QCOLOR = 3
    
    enumMap = {
        'rgb': RGB,
        'hsv+': HSV_POS,
        'hsv-': HSV_NEG,
        'clip': CLIP,
        'repeat': REPEAT,
        'mirror': MIRROR,
        'byte': BYTE,
        'float': FLOAT,
        'qcolor': QCOLOR,
    }
    
    def __init__(self, pos, color, mode=None):
        """
        ========= ==============================================================
        Arguments
        pos       Array of positions where each color is defined
        color     Array of RGBA colors.
                  Integer data types are interpreted as 0-255; float data types
                  are interpreted as 0.0-1.0
        mode      Array of color modes (ColorMap.RGB, HSV_POS, or HSV_NEG) 
                  indicating the color space that should be used when 
                  interpolating between stops. Note that the last mode value is
                  ignored. By default, the mode is entirely RGB.
        ========= ==============================================================
        """
        self.pos = pos
        self.color = color
        if mode is None:
            mode = np.ones(len(pos))
        self.mode = mode
        self.stopsCache = {}
        
    def map(self, data, mode='byte'):
        """
        Data must be either a scalar position or an array (any shape) of positions.
        """
        if isinstance(mode, basestring):
            mode = self.enumMap[mode.lower()]
            
        if mode == self.QCOLOR:
            pos, color = self.getStops(self.BYTE)
        else:
            pos, color = self.getStops(mode)
            
        data = np.clip(data, pos.min(), pos.max())
            
        if not isinstance(data, np.ndarray):
            interp = scipy.interpolate.griddata(pos, color, np.array([data]))[0]
        else:
            interp = scipy.interpolate.griddata(pos, color, data)
        
        if mode == self.QCOLOR:
            if not isinstance(data, np.ndarray):
                return QtGui.QColor(*interp)
            else:
                return [QtGui.QColor(*x) for x in interp]
        else:
            return interp
        
    def mapToQColor(self, data):
        return self.map(data, mode=self.QCOLOR)

    def mapToByte(self, data):
        return self.map(data, mode=self.BYTE)

    def mapToFloat(self, data):
        return self.map(data, mode=self.FLOAT)
    
    def getGradient(self, p1=None, p2=None):
        """Return a QLinearGradient object."""
        if p1 == None:
            p1 = QtCore.QPointF(0,0)
        if p2 == None:
            p2 = QtCore.QPointF(self.pos.max()-self.pos.min(),0)
        g = QtGui.QLinearGradient(p1, p2)
        
        pos, color = self.getStops(mode=self.BYTE)
        color = [QtGui.QColor(*x) for x in color]
        g.setStops(zip(pos, color))
        
        #if self.colorMode == 'rgb':
            #ticks = self.listTicks()
            #g.setStops([(x, QtGui.QColor(t.color)) for t,x in ticks])
        #elif self.colorMode == 'hsv':  ## HSV mode is approximated for display by interpolating 10 points between each stop
            #ticks = self.listTicks()
            #stops = []
            #stops.append((ticks[0][1], ticks[0][0].color))
            #for i in range(1,len(ticks)):
                #x1 = ticks[i-1][1]
                #x2 = ticks[i][1]
                #dx = (x2-x1) / 10.
                #for j in range(1,10):
                    #x = x1 + dx*j
                    #stops.append((x, self.getColor(x)))
                #stops.append((x2, self.getColor(x2)))
            #g.setStops(stops)
        return g
    
    def getColors(self, mode=None):
        """Return list of all colors converted to the specified mode.
        If mode is None, then no conversion is done."""
        if isinstance(mode, basestring):
            mode = self.enumMap[mode.lower()]
        
        color = self.color
        if mode in [self.BYTE, self.QCOLOR] and color.dtype.kind == 'f':
            color = (color * 255).astype(np.ubyte)
        elif mode == self.FLOAT and color.dtype.kind != 'f':
            color = color.astype(float) / 255.
            
        if mode == self.QCOLOR:
            color = [QtGui.QColor(*x) for x in color]
            
        return color
        
    def getStops(self, mode):
        ## Get fully-expanded set of RGBA stops in either float or byte mode.
        if mode not in self.stopsCache:
            color = self.color
            if mode == self.BYTE and color.dtype.kind == 'f':
                color = (color * 255).astype(np.ubyte)
            elif mode == self.FLOAT and color.dtype.kind != 'f':
                color = color.astype(float) / 255.
        
            ## to support HSV mode, we need to do a little more work..
            #stops = []
            #for i in range(len(self.pos)):
                #pos = self.pos[i]
                #color = color[i]
                
                #imode = self.mode[i]
                #if imode == self.RGB:
                    #stops.append((x,color)) 
                #else:
                    #ns = 
            self.stopsCache[mode] = (self.pos, color)
        return self.stopsCache[mode]
        
    #def getColor(self, x, toQColor=True):
        #"""
        #Return a color for a given value.
        
        #============= ==================================================================
        #**Arguments** 
        #x             Value (position on gradient) of requested color.
        #toQColor      If true, returns a QColor object, else returns a (r,g,b,a) tuple.
        #============= ==================================================================
        #"""
        #ticks = self.listTicks()
        #if x <= ticks[0][1]:
            #c = ticks[0][0].color
            #if toQColor:
                #return QtGui.QColor(c)  # always copy colors before handing them out
            #else:
                #return (c.red(), c.green(), c.blue(), c.alpha())
        #if x >= ticks[-1][1]:
            #c = ticks[-1][0].color
            #if toQColor:
                #return QtGui.QColor(c)  # always copy colors before handing them out
            #else:
                #return (c.red(), c.green(), c.blue(), c.alpha())
            
        #x2 = ticks[0][1]
        #for i in range(1,len(ticks)):
            #x1 = x2
            #x2 = ticks[i][1]
            #if x1 <= x and x2 >= x:
                #break
                
        #dx = (x2-x1)
        #if dx == 0:
            #f = 0.
        #else:
            #f = (x-x1) / dx
        #c1 = ticks[i-1][0].color
        #c2 = ticks[i][0].color
        #if self.colorMode == 'rgb':
            #r = c1.red() * (1.-f) + c2.red() * f
            #g = c1.green() * (1.-f) + c2.green() * f
            #b = c1.blue() * (1.-f) + c2.blue() * f
            #a = c1.alpha() * (1.-f) + c2.alpha() * f
            #if toQColor:
                #return QtGui.QColor(int(r), int(g), int(b), int(a))
            #else:
                #return (r,g,b,a)
        #elif self.colorMode == 'hsv':
            #h1,s1,v1,_ = c1.getHsv()
            #h2,s2,v2,_ = c2.getHsv()
            #h = h1 * (1.-f) + h2 * f
            #s = s1 * (1.-f) + s2 * f
            #v = v1 * (1.-f) + v2 * f
            #c = QtGui.QColor()
            #c.setHsv(h,s,v)
            #if toQColor:
                #return c
            #else:
                #return (c.red(), c.green(), c.blue(), c.alpha())
                    
    def getLookupTable(self, start=0.0, stop=1.0, nPts=512, alpha=None, mode='byte'):
        """
        Return an RGB(A) lookup table (ndarray). 
        
        ============= ============================================================================
        **Arguments**
        nPts           The number of points in the returned lookup table.
        alpha          True, False, or None - Specifies whether or not alpha values are included 
                       in the table. If alpha is None, it will be automatically determined.
        ============= ============================================================================
        """
        if isinstance(mode, basestring):
            mode = self.enumMap[mode.lower()]
        
        if alpha is None:
            alpha = self.usesAlpha()
            
        x = np.linspace(start, stop, nPts)
        table = self.map(x, mode)
        
        if not alpha:
            return table[:,:3]
        else:
            return table
    
    def usesAlpha(self):
        """Return True if any stops have an alpha < 255"""
        max = 1.0 if self.color.dtype.kind == 'f' else 255
        return np.any(self.color[:,3] != max)
            
    def isMapTrivial(self):
        """Return True if the gradient has exactly two stops in it: black at 0.0 and white at 1.0"""
        if len(self.pos) != 2:
            return False
        if self.pos[0] != 0.0 or self.pos[1] != 1.0:
            return False
        if self.color.dtype.kind == 'f':
            return np.all(self.color == np.array([[0.,0.,0.,1.], [1.,1.,1.,1.]]))
        else:
            return np.all(self.color == np.array([[0,0,0,255], [255,255,255,255]]))


