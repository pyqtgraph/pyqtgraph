# -*- coding: utf-8 -*-
"""
functions.py -  Miscellaneous functions with no other home
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more infomation.
"""

from .python2_3 import asUnicode
Colors = {
    'b': (0,0,255,255),
    'g': (0,255,0,255),
    'r': (255,0,0,255),
    'c': (0,255,255,255),
    'm': (255,0,255,255),
    'y': (255,255,0,255),
    'k': (0,0,0,255),
    'w': (255,255,255,255),
}  

SI_PREFIXES = asUnicode('yzafpnµm kMGTPEZY')
SI_PREFIXES_ASCII = 'yzafpnum kMGTPEZY'



from .Qt import QtGui, QtCore
import numpy as np
import decimal, re

try:
    import scipy.ndimage
    HAVE_SCIPY = True
    try:
        import scipy.weave
        USE_WEAVE = True
    except:
        USE_WEAVE = False
except ImportError:
    HAVE_SCIPY = False

from . import debug

def siScale(x, minVal=1e-25, allowUnicode=True):
    """
    Return the recommended scale factor and SI prefix string for x.
    
    Example::
    
        siScale(0.0001)   # returns (1e6, 'μ')
        # This indicates that the number 0.0001 is best represented as 0.0001 * 1e6 = 100 μUnits
    """
    
    if isinstance(x, decimal.Decimal):
        x = float(x)
        
    try:
        if np.isnan(x) or np.isinf(x):
            return(1, '')
    except:
        print(x, type(x))
        raise
    if abs(x) < minVal:
        m = 0
        x = 0
    else:
        m = int(np.clip(np.floor(np.log(abs(x))/np.log(1000)), -9.0, 9.0))
    
    if m == 0:
        pref = ''
    elif m < -8 or m > 8:
        pref = 'e%d' % (m*3)
    else:
        if allowUnicode:
            pref = SI_PREFIXES[m+8]
        else:
            pref = SI_PREFIXES_ASCII[m+8]
    p = .001**m
    
    return (p, pref)    

def siFormat(x, precision=3, suffix='', space=True, error=None, minVal=1e-25, allowUnicode=True):
    """
    Return the number x formatted in engineering notation with SI prefix.
    
    Example::
        siFormat(0.0001, suffix='V')  # returns "100 μV"
    """
    
    if space is True:
        space = ' '
    if space is False:
        space = ''
        
    
    (p, pref) = siScale(x, minVal, allowUnicode)
    if not (len(pref) > 0 and pref[0] == 'e'):
        pref = space + pref
    
    if error is None:
        fmt = "%." + str(precision) + "g%s%s"
        return fmt % (x*p, pref, suffix)
    else:
        if allowUnicode:
            plusminus = space + asUnicode("±") + space
        else:
            plusminus = " +/- "
        fmt = "%." + str(precision) + "g%s%s%s%s"
        return fmt % (x*p, pref, suffix, plusminus, siFormat(error, precision=precision, suffix=suffix, space=space, minVal=minVal))
    
def siEval(s):
    """
    Convert a value written in SI notation to its equivalent prefixless value
    
    Example::
    
        siEval("100 μV")  # returns 0.0001
    """
    
    s = asUnicode(s)
    m = re.match(r'(-?((\d+(\.\d*)?)|(\.\d+))([eE]-?\d+)?)\s*([u' + SI_PREFIXES + r']?).*$', s)
    if m is None:
        raise Exception("Can't convert string '%s' to number." % s)
    v = float(m.groups()[0])
    p = m.groups()[6]
    #if p not in SI_PREFIXES:
        #raise Exception("Can't convert string '%s' to number--unknown prefix." % s)
    if p ==  '':
        n = 0
    elif p == 'u':
        n = -2
    else:
        n = SI_PREFIXES.index(p) - 8
    return v * 1000**n
    

class Color(QtGui.QColor):
    def __init__(self, *args):
        QtGui.QColor.__init__(self, mkColor(*args))
        
    def glColor(self):
        """Return (r,g,b,a) normalized for use in opengl"""
        return (self.red()/255., self.green()/255., self.blue()/255., self.alpha()/255.)
        
    def __getitem__(self, ind):
        return (self.red, self.green, self.blue, self.alpha)[ind]()
        
    
def mkColor(*args):
    """
    Convenience function for constructing QColor from a variety of argument types. Accepted arguments are:
    
    ================ ================================================
     'c'             one of: r, g, b, c, m, y, k, w                      
     R, G, B, [A]    integers 0-255
     (R, G, B, [A])  tuple of integers 0-255
     float           greyscale, 0.0-1.0
     int             see :func:`intColor() <pyqtgraph.intColor>`
     (int, hues)     see :func:`intColor() <pyqtgraph.intColor>`
     "RGB"           hexadecimal strings; may begin with '#'
     "RGBA"          
     "RRGGBB"       
     "RRGGBBAA"     
     QColor          QColor instance; makes a copy.
    ================ ================================================
    """
    err = 'Not sure how to make a color from "%s"' % str(args)
    if len(args) == 1:
        if isinstance(args[0], QtGui.QColor):
            return QtGui.QColor(args[0])
        elif isinstance(args[0], float):
            r = g = b = int(args[0] * 255)
            a = 255
        elif isinstance(args[0], basestring):
            c = args[0]
            if c[0] == '#':
                c = c[1:]
            if len(c) == 1:
                (r, g, b, a) = Colors[c]
            if len(c) == 3:
                r = int(c[0]*2, 16)
                g = int(c[1]*2, 16)
                b = int(c[2]*2, 16)
                a = 255
            elif len(c) == 4:
                r = int(c[0]*2, 16)
                g = int(c[1]*2, 16)
                b = int(c[2]*2, 16)
                a = int(c[3]*2, 16)
            elif len(c) == 6:
                r = int(c[0:2], 16)
                g = int(c[2:4], 16)
                b = int(c[4:6], 16)
                a = 255
            elif len(c) == 8:
                r = int(c[0:2], 16)
                g = int(c[2:4], 16)
                b = int(c[4:6], 16)
                a = int(c[6:8], 16)
        elif hasattr(args[0], '__len__'):
            if len(args[0]) == 3:
                (r, g, b) = args[0]
                a = 255
            elif len(args[0]) == 4:
                (r, g, b, a) = args[0]
            elif len(args[0]) == 2:
                return intColor(*args[0])
            else:
                raise Exception(err)
        elif type(args[0]) == int:
            return intColor(args[0])
        else:
            raise Exception(err)
    elif len(args) == 3:
        (r, g, b) = args
        a = 255
    elif len(args) == 4:
        (r, g, b, a) = args
    else:
        raise Exception(err)
    
    args = [r,g,b,a]
    args = [0 if np.isnan(a) or np.isinf(a) else a for a in args]
    args = list(map(int, args))
    return QtGui.QColor(*args)


def mkBrush(*args):
    """
    | Convenience function for constructing Brush.
    | This function always constructs a solid brush and accepts the same arguments as :func:`mkColor() <pyqtgraph.mkColor>`
    | Calling mkBrush(None) returns an invisible brush.
    """
    if len(args) == 1:
        arg = args[0]
        if arg is None:
            return QtGui.QBrush(QtCore.Qt.NoBrush)
        elif isinstance(arg, QtGui.QBrush):
            return QtGui.QBrush(arg)
        else:
            color = arg
    if len(args) > 1:
        color = args
    return QtGui.QBrush(mkColor(color))

def mkPen(*args, **kargs):
    """
    Convenience function for constructing QPen. 
    
    Examples::
    
        mkPen(color)
        mkPen(color, width=2)
        mkPen(cosmetic=False, width=4.5, color='r')
        mkPen({'color': "FF0", width: 2})
        mkPen(None)   # (no pen)
    
    In these examples, *color* may be replaced with any arguments accepted by :func:`mkColor() <pyqtgraph.mkColor>`    """
    
    color = kargs.get('color', None)
    width = kargs.get('width', 1)
    style = kargs.get('style', None)
    cosmetic = kargs.get('cosmetic', True)
    hsv = kargs.get('hsv', None)
    
    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, dict):
            return mkPen(**arg)
        if isinstance(arg, QtGui.QPen):
            return QtGui.QPen(arg)  ## return a copy of this pen
        elif arg is None:
            style = QtCore.Qt.NoPen
        else:
            color = arg
    if len(args) > 1:
        color = args
        
    if color is None:
        color = mkColor(200, 200, 200)
    if hsv is not None:
        color = hsvColor(*hsv)
    else:
        color = mkColor(color)
        
    pen = QtGui.QPen(QtGui.QBrush(color), width)
    pen.setCosmetic(cosmetic)
    if style is not None:
        pen.setStyle(style)
    return pen

def hsvColor(hue, sat=1.0, val=1.0, alpha=1.0):
    """Generate a QColor from HSVa values. (all arguments are float 0.0-1.0)"""
    c = QtGui.QColor()
    c.setHsvF(hue, sat, val, alpha)
    return c

    
def colorTuple(c):
    """Return a tuple (R,G,B,A) from a QColor"""
    return (c.red(), c.green(), c.blue(), c.alpha())

def colorStr(c):
    """Generate a hex string code from a QColor"""
    return ('%02x'*4) % colorTuple(c)

def intColor(index, hues=9, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0, sat=255, alpha=255, **kargs):
    """
    Creates a QColor from a single index. Useful for stepping through a predefined list of colors.
    
    The argument *index* determines which color from the set will be returned. All other arguments determine what the set of predefined colors will be
     
    Colors are chosen by cycling across hues while varying the value (brightness). 
    By default, this selects from a list of 9 hues."""
    hues = int(hues)
    values = int(values)
    ind = int(index) % (hues * values)
    indh = ind % hues
    indv = ind / hues
    if values > 1:
        v = minValue + indv * ((maxValue-minValue) / (values-1))
    else:
        v = maxValue
    h = minHue + (indh * (maxHue-minHue)) / hues
    
    c = QtGui.QColor()
    c.setHsv(h, sat, v)
    c.setAlpha(alpha)
    return c

def glColor(*args, **kargs):
    """
    Convert a color to OpenGL color format (r,g,b,a) floats 0.0-1.0
    Accepts same arguments as :func:`mkColor <pyqtgraph.mkColor>`.
    """
    c = mkColor(*args, **kargs)
    return (c.red()/255., c.green()/255., c.blue()/255., c.alpha()/255.)

    

def makeArrowPath(headLen=20, tipAngle=20, tailLen=20, tailWidth=3, baseAngle=0):
    """
    Construct a path outlining an arrow with the given dimensions.
    The arrow points in the -x direction with tip positioned at 0,0.
    If *tipAngle* is supplied (in degrees), it overrides *headWidth*.
    If *tailLen* is None, no tail will be drawn.
    """
    headWidth = headLen * np.tan(tipAngle * 0.5 * np.pi/180.)
    path = QtGui.QPainterPath()
    path.moveTo(0,0)
    path.lineTo(headLen, -headWidth)
    if tailLen is None:
        innerY = headLen - headWidth * np.tan(baseAngle*np.pi/180.)
        path.lineTo(innerY, 0)
    else:
        tailWidth *= 0.5
        innerY = headLen - (headWidth-tailWidth) * np.tan(baseAngle*np.pi/180.)
        path.lineTo(innerY, -tailWidth)
        path.lineTo(headLen + tailLen, -tailWidth)
        path.lineTo(headLen + tailLen, tailWidth)
        path.lineTo(innerY, tailWidth)
    path.lineTo(headLen, headWidth)
    path.lineTo(0,0)
    return path
    
    
    
def affineSlice(data, shape, origin, vectors, axes, order=1, returnCoords=False, **kargs):
    """
    Take a slice of any orientation through an array. This is useful for extracting sections of multi-dimensional arrays such as MRI images for viewing as 1D or 2D data.
    
    The slicing axes are aribtrary; they do not need to be orthogonal to the original data or even to each other. It is possible to use this function to extract arbitrary linear, rectangular, or parallelepiped shapes from within larger datasets. The original data is interpolated onto a new array of coordinates using scipy.ndimage.map_coordinates (see the scipy documentation for more information about this).
    
    For a graphical interface to this function, see :func:`ROI.getArrayRegion <pyqtgraph.ROI.getArrayRegion>`
    
    ==============  ====================================================================================================
    Arguments:
    *data*          (ndarray) the original dataset
    *shape*         the shape of the slice to take (Note the return value may have more dimensions than len(shape))
    *origin*        the location in the original dataset that will become the origin of the sliced data.
    *vectors*       list of unit vectors which point in the direction of the slice axes. Each vector must have the same 
                    length as *axes*. If the vectors are not unit length, the result will be scaled relative to the 
                    original data. If the vectors are not orthogonal, the result will be sheared relative to the 
                    original data.
    *axes*          The axes in the original dataset which correspond to the slice *vectors*
    *order*         The order of spline interpolation. Default is 1 (linear). See scipy.ndimage.map_coordinates
                    for more information.
    *returnCoords*  If True, return a tuple (result, coords) where coords is the array of coordinates used to select
                    values from the original dataset.
    *All extra keyword arguments are passed to scipy.ndimage.map_coordinates.*
    --------------------------------------------------------------------------------------------------------------------
    ==============  ====================================================================================================
    
    Note the following must be true: 
        
        | len(shape) == len(vectors) 
        | len(origin) == len(axes) == len(vectors[i])
        
    Example: start with a 4D fMRI data set, take a diagonal-planar slice out of the last 3 axes
        
        * data = array with dims (time, x, y, z) = (100, 40, 40, 40)
        * The plane to pull out is perpendicular to the vector (x,y,z) = (1,1,1) 
        * The origin of the slice will be at (x,y,z) = (40, 0, 0)
        * We will slice a 20x20 plane from each timepoint, giving a final shape (100, 20, 20)
        
    The call for this example would look like::
        
        affineSlice(data, shape=(20,20), origin=(40,0,0), vectors=((-1, 1, 0), (-1, 0, 1)), axes=(1,2,3))
    
    """
    if not HAVE_SCIPY:
        raise Exception("This function requires the scipy library, but it does not appear to be importable.")

    # sanity check
    if len(shape) != len(vectors):
        raise Exception("shape and vectors must have same length.")
    if len(origin) != len(axes):
        raise Exception("origin and axes must have same length.")
    for v in vectors:
        if len(v) != len(axes):
            raise Exception("each vector must be same length as axes.")
        
    shape = list(map(np.ceil, shape))

    ## transpose data so slice axes come first
    trAx = list(range(data.ndim))
    for x in axes:
        trAx.remove(x)
    tr1 = tuple(axes) + tuple(trAx)
    data = data.transpose(tr1)
    #print "tr1:", tr1
    ## dims are now [(slice axes), (other axes)]
    

    ## make sure vectors are arrays
    vectors = np.array(vectors)
    origin = np.array(origin)
    origin.shape = (len(axes),) + (1,)*len(shape)
    
    ## Build array of sample locations. 
    grid = np.mgrid[tuple([slice(0,x) for x in shape])]  ## mesh grid of indexes
    #print shape, grid.shape
    x = (grid[np.newaxis,...] * vectors.transpose()[(Ellipsis,) + (np.newaxis,)*len(shape)]).sum(axis=1)  ## magic
    x += origin
    #print "X values:"
    #print x
    ## iterate manually over unused axes since map_coordinates won't do it for us
    extraShape = data.shape[len(axes):]
    output = np.empty(tuple(shape) + extraShape, dtype=data.dtype)
    for inds in np.ndindex(*extraShape):
        ind = (Ellipsis,) + inds
        #print data[ind].shape, x.shape, output[ind].shape, output.shape
        output[ind] = scipy.ndimage.map_coordinates(data[ind], x, order=order, **kargs)
    
    tr = list(range(output.ndim))
    trb = []
    for i in range(min(axes)):
        ind = tr1.index(i) + (len(shape)-len(axes))
        tr.remove(ind)
        trb.append(ind)
    tr2 = tuple(trb+tr)

    ## Untranspose array before returning
    output = output.transpose(tr2)
    if returnCoords:
        return (output, x)
    else:
        return output

def transformToArray(tr):
    """
    Given a QTransform, return a 3x3 numpy array.
    Given a QMatrix4x4, return a 4x4 numpy array.
    
    Example: map an array of x,y coordinates through a transform::
    
        ## coordinates to map are (1,5), (2,6), (3,7), and (4,8)
        coords = np.array([[1,2,3,4], [5,6,7,8], [1,1,1,1]])  # the extra '1' coordinate is needed for translation to work
        
        ## Make an example transform
        tr = QtGui.QTransform()
        tr.translate(3,4)
        tr.scale(2, 0.1)
        
        ## convert to array
        m = pg.transformToArray()[:2]  # ignore the perspective portion of the transformation
        
        ## map coordinates through transform
        mapped = np.dot(m, coords)
    """
    #return np.array([[tr.m11(), tr.m12(), tr.m13()],[tr.m21(), tr.m22(), tr.m23()],[tr.m31(), tr.m32(), tr.m33()]])
    ## The order of elements given by the method names m11..m33 is misleading--
    ## It is most common for x,y translation to occupy the positions 1,3 and 2,3 in
    ## a transformation matrix. However, with QTransform these values appear at m31 and m32.
    ## So the correct interpretation is transposed:
    if isinstance(tr, QtGui.QTransform):
        return np.array([[tr.m11(), tr.m21(), tr.m31()], [tr.m12(), tr.m22(), tr.m32()], [tr.m13(), tr.m23(), tr.m33()]])
    elif isinstance(tr, QtGui.QMatrix4x4):
        return np.array(tr.copyDataTo()).reshape(4,4)
    else:
        raise Exception("Transform argument must be either QTransform or QMatrix4x4.")

def transformCoordinates(tr, coords):
    """
    Map a set of 2D or 3D coordinates through a QTransform or QMatrix4x4.
    The shape of coords must be (2,...) or (3,...)
    The mapping will _ignore_ any perspective transformations.
    """
    nd = coords.shape[0]
    m = transformToArray(tr)    
    m = m[:m.shape[0]-1]  # remove perspective
    
    ## If coords are 3D and tr is 2D, assume no change for Z axis
    if m.shape == (2,3) and nd == 3:
        m2 = np.zeros((3,4))
        m2[:2, :2] = m[:2,:2]
        m2[:2, 3] = m[:2,2]
        m2[2,2] = 1
        m = m2
    
    ## if coords are 2D and tr is 3D, ignore Z axis
    if m.shape == (3,4) and nd == 2:
        m2 = np.empty((2,3))
        m2[:,:2] = m[:2,:2]
        m2[:,2] = m[:2,3]
        m = m2
    
    ## reshape tr and coords to prepare for multiplication
    m = m.reshape(m.shape + (1,)*(coords.ndim-1))
    coords = coords[np.newaxis, ...]
    
    # separate scale/rotate and translation    
    translate = m[:,-1]  
    m = m[:, :-1]
    
    ## map coordinates and return
    mapped = (m*coords).sum(axis=0)  ## apply scale/rotate
    mapped += translate
    return mapped
    
    
def solve3DTransform(points1, points2):
    """
    Find a 3D transformation matrix that maps points1 onto points2
    points must be specified as a list of 4 Vectors.
    """
    if not HAVE_SCIPY:
        raise Exception("This function depends on the scipy library, but it does not appear to be importable.")
    A = np.array([[points1[i].x(), points1[i].y(), points1[i].z(), 1] for i in range(4)])
    B = np.array([[points2[i].x(), points2[i].y(), points2[i].z(), 1] for i in range(4)])
    
    ## solve 3 sets of linear equations to determine transformation matrix elements
    matrix = np.zeros((4,4))
    for i in range(3):
        matrix[i] = scipy.linalg.solve(A, B[:,i])  ## solve Ax = B; x is one row of the desired transformation matrix
    
    return matrix
    
def solveBilinearTransform(points1, points2):
    """
    Find a bilinear transformation matrix (2x4) that maps points1 onto points2
    points must be specified as a list of 4 Vector, Point, QPointF, etc.
    
    To use this matrix to map a point [x,y]::
    
        mapped = np.dot(matrix, [x*y, x, y, 1])
    """
    if not HAVE_SCIPY:
        raise Exception("This function depends on the scipy library, but it does not appear to be importable.")
    ## A is 4 rows (points) x 4 columns (xy, x, y, 1)
    ## B is 4 rows (points) x 2 columns (x, y)
    A = np.array([[points1[i].x()*points1[i].y(), points1[i].x(), points1[i].y(), 1] for i in range(4)])
    B = np.array([[points2[i].x(), points2[i].y()] for i in range(4)])
    
    ## solve 2 sets of linear equations to determine transformation matrix elements
    matrix = np.zeros((2,4))
    for i in range(2):
        matrix[i] = scipy.linalg.solve(A, B[:,i])  ## solve Ax = B; x is one row of the desired transformation matrix
    
    return matrix
    
    
    
    

def makeARGB(data, lut=None, levels=None, useRGBA=False): 
    """ 
    Convert a 2D or 3D array into an ARGB array suitable for building QImages
    Will optionally do scaling and/or table lookups to determine final colors.
    
    Returns the ARGB array (values 0-255) and a boolean indicating whether there is alpha channel data.
    
    Arguments:
        data  - 2D or 3D numpy array of int/float types
        
                For 2D arrays (x, y):
                  * The color will be determined using a lookup table (see argument 'lut').
                  * If levels are given, the data is rescaled and converted to int
                    before using the lookup table.
                 
                For 3D arrays (x, y, rgba):
                  * The third axis must have length 3 or 4 and will be interpreted as RGBA.
                  * The 'lut' argument is not allowed.
                 
        lut   - Lookup table for 2D data. May be 1D or 2D (N,rgba) and must have dtype=ubyte.
                Values in data will be converted to color by indexing directly from lut.
                Lookup tables can be built using GradientWidget.
        levels - List [min, max]; optionally rescale data before converting through the
                lookup table.   rescaled = (data-min) * len(lut) / (max-min)
        useRGBA - If True, the data is returned in RGBA order. The default is 
                  False, which returns in BGRA order for use with QImage.
                
    """
    prof = debug.Profiler('functions.makeARGB', disabled=True)
    
    ## sanity checks
    if data.ndim == 3:
        if data.shape[2] not in (3,4):
            raise Exception("data.shape[2] must be 3 or 4")
        #if lut is not None:
            #raise Exception("can not use lookup table with 3D data")
    elif data.ndim != 2:
        raise Exception("data must be 2D or 3D")
        
    if lut is not None:
        if lut.ndim == 2:
            if lut.shape[1] not in (3,4):
                raise Exception("lut.shape[1] must be 3 or 4")
        elif lut.ndim != 1:
            raise Exception("lut must be 1D or 2D")
        if lut.dtype != np.ubyte:
            raise Exception('lookup table must have dtype=ubyte (got %s instead)' % str(lut.dtype))

    if levels is not None:
        levels = np.array(levels)
        if levels.shape == (2,):
            pass
        elif levels.shape in [(3,2), (4,2)]:
            if data.ndim == 3:
                raise Exception("Can not use 2D levels with 3D data.")
            if lut is not None:
                raise Exception('Can not use 2D levels and lookup table together.')
        else:
            raise Exception("Levels must have shape (2,) or (3,2) or (4,2)")
        
    prof.mark('1')

    if lut is not None:
        lutLength = lut.shape[0]
    else:
        lutLength = 256

    ## weave requires contiguous arrays
    global USE_WEAVE
    if (levels is not None or lut is not None) and USE_WEAVE:
        data = np.ascontiguousarray(data)

    ## Apply levels if given
    if levels is not None:
        
        try:  ## use weave to speed up scaling
            if not USE_WEAVE:
                raise Exception('Weave is disabled; falling back to slower version.')
            if levels.ndim == 1:
                scale = float(lutLength) / (levels[1]-levels[0])
                offset = float(levels[0])
                data = rescaleData(data, scale, offset)
            else:
                if data.ndim == 2:
                    newData = np.empty(data.shape+(levels.shape[0],), dtype=np.uint32)
                    for i in range(levels.shape[0]):
                        scale = float(lutLength / (levels[i,1]-levels[i,0]))
                        offset = float(levels[i,0])
                        newData[...,i] = rescaleData(data, scale, offset)
                elif data.ndim == 3:
                    newData = np.empty(data.shape, dtype=np.uint32)
                    for i in range(data.shape[2]):
                        scale = float(lutLength / (levels[i,1]-levels[i,0]))
                        offset = float(levels[i,0])
                        #print scale, offset, data.shape, newData.shape, levels.shape
                        newData[...,i] = rescaleData(data[...,i], scale, offset)
                data = newData
        except:
            if USE_WEAVE:
                debug.printExc("Error; disabling weave.")
                USE_WEAVE = False
            
            if levels.ndim == 1:
                if data.ndim == 2:
                    levels = levels[np.newaxis, np.newaxis, :]
                else:
                    levels = levels[np.newaxis, np.newaxis, np.newaxis, :]
            else:
                levels = levels[np.newaxis, np.newaxis, ...]
                if data.ndim == 2:
                    data = data[..., np.newaxis]
            data = ((data-levels[...,0]) * lutLength) / (levels[...,1]-levels[...,0])
        
    prof.mark('2')


    ## apply LUT if given
    if lut is not None and data.ndim == 2:
        
        if data.dtype.kind not in ('i', 'u'):
            data = data.astype(int)
            
        data = np.clip(data, 0, lutLength-1)
        try:
            if not USE_WEAVE:
                raise Exception('Weave is disabled; falling back to slower version.')
            
            newData = np.empty((data.size,) + lut.shape[1:], dtype=np.uint8)
            flat = data.reshape(data.size)
            size = data.size
            ncol = lut.shape[1]
            newStride = newData.strides[0]
            newColStride = newData.strides[1]
            lutStride = lut.strides[0]
            lutColStride = lut.strides[1]
            flatStride = flat.strides[0] / flat.dtype.itemsize
            
            #print "newData:", newData.shape, newData.dtype
            #print "flat:", flat.shape, flat.dtype, flat.min(), flat.max()
            #print "lut:", lut.shape, lut.dtype
            #print "size:", size, "ncols:", ncol
            #print "strides:", newStride, newColStride, lutStride, lutColStride, flatStride
            
            code = """
            
            for( int i=0; i<size; i++ ) {
                for( int j=0; j<ncol; j++ ) {
                    newData[i*newStride + j*newColStride] = lut[flat[i*flatStride]*lutStride + j*lutColStride];
                }
            }
            """
            scipy.weave.inline(code, ['flat', 'lut', 'newData', 'size', 'ncol', 'newStride', 'lutStride', 'flatStride', 'newColStride', 'lutColStride'])
            data = newData.reshape(data.shape + lut.shape[1:])
        except:
            if USE_WEAVE:
                debug.printExc("Error; disabling weave.")
                USE_WEAVE = False
            data = lut[data]
    else:
        if data.dtype is not np.ubyte:
            data = np.clip(data, 0, 255).astype(np.ubyte)

    prof.mark('3')


    ## copy data into ARGB ordered array
    imgData = np.empty(data.shape[:2]+(4,), dtype=np.ubyte)
    if data.ndim == 2:
        data = data[..., np.newaxis]

    prof.mark('4')

    if useRGBA:
        order = [0,1,2,3] ## array comes out RGBA
    else:
        order = [2,1,0,3] ## for some reason, the colors line up as BGR in the final image.
        
    if data.shape[2] == 1:
        for i in range(3):
            imgData[..., order[i]] = data[..., 0]    
    else:
        for i in range(0, data.shape[2]):
            imgData[..., order[i]] = data[..., i]    
        
    prof.mark('5')
        
    if data.shape[2] == 4:
        alpha = True
    else:
        alpha = False
        imgData[..., 3] = 255
        
    prof.mark('6')
        
    prof.finish()
    return imgData, alpha
    

def makeQImage(imgData, alpha):
    """Turn an ARGB array into QImage"""
    ## create QImage from buffer
    prof = debug.Profiler('functions.makeQImage', disabled=True)
    
    if alpha:
        imgFormat = QtGui.QImage.Format_ARGB32
    else:
        imgFormat = QtGui.QImage.Format_RGB32
        
    imgData = imgData.transpose((1, 0, 2))  ## QImage expects the row/column order to be opposite
    try:
        buf = imgData.data
    except AttributeError:  ## happens when image data is non-contiguous
        imgData = np.ascontiguousarray(imgData)
        buf = imgData.data
        
    prof.mark('1')
    qimage = QtGui.QImage(buf, imgData.shape[1], imgData.shape[0], imgFormat)
    prof.mark('2')
    qimage.data = imgData
    prof.finish()
    return qimage


def rescaleData(data, scale, offset):
    newData = np.empty((data.size,), dtype=np.int)
    flat = data.reshape(data.size)
    size = data.size
    
    code = """
    double sc = (double)scale;
    double off = (double)offset;
    for( int i=0; i<size; i++ ) {
        newData[i] = (int)(((double)flat[i] - off) * sc);
    }
    """
    scipy.weave.inline(code, ['flat', 'newData', 'size', 'offset', 'scale'], compiler='gcc')
    data = newData.reshape(data.shape)
    return data
    

#def isosurface(data, level):
    #"""
    #Generate isosurface from volumetric data using marching tetrahedra algorithm.
    #See Paul Bourke, "Polygonising a Scalar Field Using Tetrahedrons"  (http://local.wasp.uwa.edu.au/~pbourke/geometry/polygonise/)
    
    #*data*   3D numpy array of scalar values
    #*level*  The level at which to generate an isosurface
    #"""
    
    #facets = []
    
    ### mark everything below the isosurface level
    #mask = data < level
    
    #### make eight sub-fields 
    #fields = np.empty((2,2,2), dtype=object)
    #slices = [slice(0,-1), slice(1,None)]
    #for i in [0,1]:
        #for j in [0,1]:
            #for k in [0,1]:
                #fields[i,j,k] = mask[slices[i], slices[j], slices[k]]
    
    
    
    ### split each cell into 6 tetrahedra
    ### these all have the same 'orienation'; points 1,2,3 circle 
    ### clockwise around point 0
    #tetrahedra = [
        #[(0,1,0), (1,1,1), (0,1,1), (1,0,1)],
        #[(0,1,0), (0,1,1), (0,0,1), (1,0,1)],
        #[(0,1,0), (0,0,1), (0,0,0), (1,0,1)],
        #[(0,1,0), (0,0,0), (1,0,0), (1,0,1)],
        #[(0,1,0), (1,0,0), (1,1,0), (1,0,1)],
        #[(0,1,0), (1,1,0), (1,1,1), (1,0,1)]
    #]
    
    ### each tetrahedron will be assigned an index
    ### which determines how to generate its facets.
    ### this structure is: 
    ###    facets[index][facet1, facet2, ...]
    ### where each facet is triangular and its points are each 
    ### interpolated between two points on the tetrahedron
    ###    facet = [(p1a, p1b), (p2a, p2b), (p3a, p3b)]
    ### facet points always circle clockwise if you are looking 
    ### at them from below the isosurface.
    #indexFacets = [
        #[],  ## all above
        #[[(0,1), (0,2), (0,3)]],  # 0 below
        #[[(1,0), (1,3), (1,2)]],   # 1 below
        #[[(0,2), (1,3), (1,2)], [(0,2), (0,3), (1,3)]],   # 0,1 below
        #[[(2,0), (2,1), (2,3)]],   # 2 below
        #[[(0,3), (1,2), (2,3)], [(0,3), (0,1), (1,2)]],   # 0,2 below
        #[[(1,0), (2,3), (2,0)], [(1,0), (1,3), (2,3)]],   # 1,2 below
        #[[(3,0), (3,1), (3,2)]],   # 3 above
        #[[(3,0), (3,2), (3,1)]],   # 3 below
        #[[(1,0), (2,0), (2,3)], [(1,0), (2,3), (1,3)]],   # 0,3 below
        #[[(0,3), (2,3), (1,2)], [(0,3), (1,2), (0,1)]],   # 1,3 below
        #[[(2,0), (2,3), (2,1)]], # 0,1,3 below
        #[[(0,2), (1,2), (1,3)], [(0,2), (1,3), (0,3)]],   # 2,3 below
        #[[(1,0), (1,2), (1,3)]], # 0,2,3 below
        #[[(0,1), (0,3), (0,2)]], # 1,2,3 below
        #[]  ## all below
    #]
    
    #for tet in tetrahedra:
        
        ### get the 4 fields for this tetrahedron
        #tetFields = [fields[c] for c in tet]
        
        ### generate an index for each grid cell
        #index = tetFields[0] + tetFields[1]*2 + tetFields[2]*4 + tetFields[3]*8
        
        ### add facets
        #for i in xrange(index.shape[0]):                 # data x-axis
            #for j in xrange(index.shape[1]):             # data y-axis
                #for k in xrange(index.shape[2]):         # data z-axis
                    #for f in indexFacets[index[i,j,k]]:  # faces to generate for this tet
                        #pts = []
                        #for l in [0,1,2]:      # points in this face
                            #p1 = tet[f[l][0]]  # tet corner 1
                            #p2 = tet[f[l][1]]  # tet corner 2
                            #pts.append([(p1[x]+p2[x])*0.5+[i,j,k][x]+0.5 for x in [0,1,2]]) ## interpolate between tet corners
                        #facets.append(pts)

    #return facets
    

def isocurve(data, level):
    """
    Generate isocurve from 2D data using marching squares algorithm.
    
    *data*   2D numpy array of scalar values
    *level*  The level at which to generate an isosurface
    
    This function is SLOW; plenty of room for optimization here.
    """    
    
    sideTable = [
    [],
    [0,1],
    [1,2],
    [0,2],
    [0,3],
    [1,3],
    [0,1,2,3],
    [2,3],
    [2,3],
    [0,1,2,3],
    [1,3],
    [0,3],
    [0,2],
    [1,2],
    [0,1],
    []
    ]
    
    edgeKey=[
    [(0,1),(0,0)],
    [(0,0), (1,0)],
    [(1,0), (1,1)],
    [(1,1), (0,1)]
    ]
    
    
    lines = []
    
    ## mark everything below the isosurface level
    mask = data < level
    
    ### make four sub-fields and compute indexes for grid cells
    index = np.zeros([x-1 for x in data.shape], dtype=np.ubyte)
    fields = np.empty((2,2), dtype=object)
    slices = [slice(0,-1), slice(1,None)]
    for i in [0,1]:
        for j in [0,1]:
            fields[i,j] = mask[slices[i], slices[j]]
            #vertIndex = i - 2*j*i + 3*j + 4*k  ## this is just to match Bourk's vertex numbering scheme
            vertIndex = i+2*j
            #print i,j,k," : ", fields[i,j,k], 2**vertIndex
            index += fields[i,j] * 2**vertIndex
            #print index
    #print index
    
    ## add lines
    for i in range(index.shape[0]):                 # data x-axis
        for j in range(index.shape[1]):             # data y-axis     
            sides = sideTable[index[i,j]]
            for l in range(0, len(sides), 2):     ## faces for this grid cell
                edges = sides[l:l+2]
                pts = []
                for m in [0,1]:      # points in this face
                    p1 = edgeKey[edges[m]][0] # p1, p2 are points at either side of an edge
                    p2 = edgeKey[edges[m]][1]
                    v1 = data[i+p1[0], j+p1[1]] # v1 and v2 are the values at p1 and p2
                    v2 = data[i+p2[0], j+p2[1]]
                    f = (level-v1) / (v2-v1)
                    fi = 1.0 - f
                    p = (    ## interpolate between corners
                        p1[0]*fi + p2[0]*f + i + 0.5, 
                        p1[1]*fi + p2[1]*f + j + 0.5
                        )
                    pts.append(p)
                lines.append(pts)

    return lines ## a list of pairs of points
    
    
def isosurface(data, level):
    """
    Generate isosurface from volumetric data using marching cubes algorithm.
    See Paul Bourke, "Polygonising a Scalar Field"  
    (http://local.wasp.uwa.edu.au/~pbourke/geometry/polygonise/)
    
    *data*   3D numpy array of scalar values
    *level*  The level at which to generate an isosurface
    
    Returns a list of faces; each face is a list of three vertexes and each vertex is a tuple of three floats.
    
    This function is SLOW; plenty of room for optimization here.
    """

    ## map from grid cell index to edge index.
    ## grid cell index tells us which corners are below the isosurface,
    ## edge index tells us which edges are cut by the isosurface.
    ## (Data stolen from Bourk; see above.)
    edgeTable = [
    0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0   ]

    ## Table of triangles to use for filling each grid cell.
    ## Each set of three integers tells us which three edges to
    ## draw a triangle between.
    ## (Data stolen from Bourk; see above.)
    triTable = [
        [],
        [0, 8, 3],
        [0, 1, 9],
        [1, 8, 3, 9, 8, 1],
        [1, 2, 10],
        [0, 8, 3, 1, 2, 10],
        [9, 2, 10, 0, 2, 9],
        [2, 8, 3, 2, 10, 8, 10, 9, 8],
        [3, 11, 2],
        [0, 11, 2, 8, 11, 0],
        [1, 9, 0, 2, 3, 11],
        [1, 11, 2, 1, 9, 11, 9, 8, 11],
        [3, 10, 1, 11, 10, 3],
        [0, 10, 1, 0, 8, 10, 8, 11, 10],
        [3, 9, 0, 3, 11, 9, 11, 10, 9],
        [9, 8, 10, 10, 8, 11],
        [4, 7, 8],
        [4, 3, 0, 7, 3, 4],
        [0, 1, 9, 8, 4, 7],
        [4, 1, 9, 4, 7, 1, 7, 3, 1],
        [1, 2, 10, 8, 4, 7],
        [3, 4, 7, 3, 0, 4, 1, 2, 10],
        [9, 2, 10, 9, 0, 2, 8, 4, 7],
        [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4],
        [8, 4, 7, 3, 11, 2],
        [11, 4, 7, 11, 2, 4, 2, 0, 4],
        [9, 0, 1, 8, 4, 7, 2, 3, 11],
        [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1],
        [3, 10, 1, 3, 11, 10, 7, 8, 4],
        [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4],
        [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3],
        [4, 7, 11, 4, 11, 9, 9, 11, 10],
        [9, 5, 4],
        [9, 5, 4, 0, 8, 3],
        [0, 5, 4, 1, 5, 0],
        [8, 5, 4, 8, 3, 5, 3, 1, 5],
        [1, 2, 10, 9, 5, 4],
        [3, 0, 8, 1, 2, 10, 4, 9, 5],
        [5, 2, 10, 5, 4, 2, 4, 0, 2],
        [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8],
        [9, 5, 4, 2, 3, 11],
        [0, 11, 2, 0, 8, 11, 4, 9, 5],
        [0, 5, 4, 0, 1, 5, 2, 3, 11],
        [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5],
        [10, 3, 11, 10, 1, 3, 9, 5, 4],
        [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10],
        [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3],
        [5, 4, 8, 5, 8, 10, 10, 8, 11],
        [9, 7, 8, 5, 7, 9],
        [9, 3, 0, 9, 5, 3, 5, 7, 3],
        [0, 7, 8, 0, 1, 7, 1, 5, 7],
        [1, 5, 3, 3, 5, 7],
        [9, 7, 8, 9, 5, 7, 10, 1, 2],
        [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3],
        [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2],
        [2, 10, 5, 2, 5, 3, 3, 5, 7],
        [7, 9, 5, 7, 8, 9, 3, 11, 2],
        [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11],
        [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7],
        [11, 2, 1, 11, 1, 7, 7, 1, 5],
        [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11],
        [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0],
        [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0],
        [11, 10, 5, 7, 11, 5],
        [10, 6, 5],
        [0, 8, 3, 5, 10, 6],
        [9, 0, 1, 5, 10, 6],
        [1, 8, 3, 1, 9, 8, 5, 10, 6],
        [1, 6, 5, 2, 6, 1],
        [1, 6, 5, 1, 2, 6, 3, 0, 8],
        [9, 6, 5, 9, 0, 6, 0, 2, 6],
        [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8],
        [2, 3, 11, 10, 6, 5],
        [11, 0, 8, 11, 2, 0, 10, 6, 5],
        [0, 1, 9, 2, 3, 11, 5, 10, 6],
        [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11],
        [6, 3, 11, 6, 5, 3, 5, 1, 3],
        [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6],
        [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9],
        [6, 5, 9, 6, 9, 11, 11, 9, 8],
        [5, 10, 6, 4, 7, 8],
        [4, 3, 0, 4, 7, 3, 6, 5, 10],
        [1, 9, 0, 5, 10, 6, 8, 4, 7],
        [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4],
        [6, 1, 2, 6, 5, 1, 4, 7, 8],
        [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7],
        [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6],
        [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9],
        [3, 11, 2, 7, 8, 4, 10, 6, 5],
        [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11],
        [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6],
        [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6],
        [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6],
        [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11],
        [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7],
        [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9],
        [10, 4, 9, 6, 4, 10],
        [4, 10, 6, 4, 9, 10, 0, 8, 3],
        [10, 0, 1, 10, 6, 0, 6, 4, 0],
        [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10],
        [1, 4, 9, 1, 2, 4, 2, 6, 4],
        [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4],
        [0, 2, 4, 4, 2, 6],
        [8, 3, 2, 8, 2, 4, 4, 2, 6],
        [10, 4, 9, 10, 6, 4, 11, 2, 3],
        [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6],
        [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10],
        [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1],
        [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3],
        [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1],
        [3, 11, 6, 3, 6, 0, 0, 6, 4],
        [6, 4, 8, 11, 6, 8],
        [7, 10, 6, 7, 8, 10, 8, 9, 10],
        [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10],
        [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0],
        [10, 6, 7, 10, 7, 1, 1, 7, 3],
        [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7],
        [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9],
        [7, 8, 0, 7, 0, 6, 6, 0, 2],
        [7, 3, 2, 6, 7, 2],
        [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7],
        [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7],
        [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11],
        [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1],
        [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6],
        [0, 9, 1, 11, 6, 7],
        [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0],
        [7, 11, 6],
        [7, 6, 11],
        [3, 0, 8, 11, 7, 6],
        [0, 1, 9, 11, 7, 6],
        [8, 1, 9, 8, 3, 1, 11, 7, 6],
        [10, 1, 2, 6, 11, 7],
        [1, 2, 10, 3, 0, 8, 6, 11, 7],
        [2, 9, 0, 2, 10, 9, 6, 11, 7],
        [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8],
        [7, 2, 3, 6, 2, 7],
        [7, 0, 8, 7, 6, 0, 6, 2, 0],
        [2, 7, 6, 2, 3, 7, 0, 1, 9],
        [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6],
        [10, 7, 6, 10, 1, 7, 1, 3, 7],
        [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8],
        [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7],
        [7, 6, 10, 7, 10, 8, 8, 10, 9],
        [6, 8, 4, 11, 8, 6],
        [3, 6, 11, 3, 0, 6, 0, 4, 6],
        [8, 6, 11, 8, 4, 6, 9, 0, 1],
        [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6],
        [6, 8, 4, 6, 11, 8, 2, 10, 1],
        [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6],
        [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9],
        [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3],
        [8, 2, 3, 8, 4, 2, 4, 6, 2],
        [0, 4, 2, 4, 6, 2],
        [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8],
        [1, 9, 4, 1, 4, 2, 2, 4, 6],
        [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1],
        [10, 1, 0, 10, 0, 6, 6, 0, 4],
        [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3],
        [10, 9, 4, 6, 10, 4],
        [4, 9, 5, 7, 6, 11],
        [0, 8, 3, 4, 9, 5, 11, 7, 6],
        [5, 0, 1, 5, 4, 0, 7, 6, 11],
        [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5],
        [9, 5, 4, 10, 1, 2, 7, 6, 11],
        [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5],
        [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2],
        [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6],
        [7, 2, 3, 7, 6, 2, 5, 4, 9],
        [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7],
        [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0],
        [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8],
        [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7],
        [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4],
        [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10],
        [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10],
        [6, 9, 5, 6, 11, 9, 11, 8, 9],
        [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5],
        [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11],
        [6, 11, 3, 6, 3, 5, 5, 3, 1],
        [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6],
        [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10],
        [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5],
        [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3],
        [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2],
        [9, 5, 6, 9, 6, 0, 0, 6, 2],
        [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8],
        [1, 5, 6, 2, 1, 6],
        [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6],
        [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0],
        [0, 3, 8, 5, 6, 10],
        [10, 5, 6],
        [11, 5, 10, 7, 5, 11],
        [11, 5, 10, 11, 7, 5, 8, 3, 0],
        [5, 11, 7, 5, 10, 11, 1, 9, 0],
        [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1],
        [11, 1, 2, 11, 7, 1, 7, 5, 1],
        [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11],
        [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7],
        [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2],
        [2, 5, 10, 2, 3, 5, 3, 7, 5],
        [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5],
        [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2],
        [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2],
        [1, 3, 5, 3, 7, 5],
        [0, 8, 7, 0, 7, 1, 1, 7, 5],
        [9, 0, 3, 9, 3, 5, 5, 3, 7],
        [9, 8, 7, 5, 9, 7],
        [5, 8, 4, 5, 10, 8, 10, 11, 8],
        [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0],
        [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5],
        [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4],
        [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8],
        [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11],
        [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5],
        [9, 4, 5, 2, 11, 3],
        [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4],
        [5, 10, 2, 5, 2, 4, 4, 2, 0],
        [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9],
        [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2],
        [8, 4, 5, 8, 5, 3, 3, 5, 1],
        [0, 4, 5, 1, 0, 5],
        [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5],
        [9, 4, 5],
        [4, 11, 7, 4, 9, 11, 9, 10, 11],
        [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11],
        [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11],
        [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4],
        [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2],
        [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3],
        [11, 7, 4, 11, 4, 2, 2, 4, 0],
        [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4],
        [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9],
        [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7],
        [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10],
        [1, 10, 2, 8, 7, 4],
        [4, 9, 1, 4, 1, 7, 7, 1, 3],
        [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1],
        [4, 0, 3, 7, 4, 3],
        [4, 8, 7],
        [9, 10, 8, 10, 11, 8],
        [3, 0, 9, 3, 9, 11, 11, 9, 10],
        [0, 1, 10, 0, 10, 8, 8, 10, 11],
        [3, 1, 10, 11, 3, 10],
        [1, 2, 11, 1, 11, 9, 9, 11, 8],
        [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9],
        [0, 2, 11, 8, 0, 11],
        [3, 2, 11],
        [2, 3, 8, 2, 8, 10, 10, 8, 9],
        [9, 10, 2, 0, 9, 2],
        [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8],
        [1, 10, 2],
        [1, 3, 8, 9, 1, 8],
        [0, 9, 1],
        [0, 3, 8],
        []
    ]    
    
    ## translation between edge index and 
    ## the vertex indexes that bound the edge
    edgeKey = [
        [(0,0,0), (1,0,0)],
        [(1,0,0), (1,1,0)],
        [(1,1,0), (0,1,0)],
        [(0,1,0), (0,0,0)],
        [(0,0,1), (1,0,1)],
        [(1,0,1), (1,1,1)],
        [(1,1,1), (0,1,1)],
        [(0,1,1), (0,0,1)],
        [(0,0,0), (0,0,1)],
        [(1,0,0), (1,0,1)],
        [(1,1,0), (1,1,1)],
        [(0,1,0), (0,1,1)],
    ]
    
    
    
    facets = []
    
    ## mark everything below the isosurface level
    mask = data < level
    
    ### make eight sub-fields and compute indexes for grid cells
    index = np.zeros([x-1 for x in data.shape], dtype=np.ubyte)
    fields = np.empty((2,2,2), dtype=object)
    slices = [slice(0,-1), slice(1,None)]
    for i in [0,1]:
        for j in [0,1]:
            for k in [0,1]:
                fields[i,j,k] = mask[slices[i], slices[j], slices[k]]
                vertIndex = i - 2*j*i + 3*j + 4*k  ## this is just to match Bourk's vertex numbering scheme
                #print i,j,k," : ", fields[i,j,k], 2**vertIndex
                index += fields[i,j,k] * 2**vertIndex
                #print index
    #print index
    
    ## add facets
    for i in range(index.shape[0]):                 # data x-axis
        for j in range(index.shape[1]):             # data y-axis
            for k in range(index.shape[2]):         # data z-axis
                tris = triTable[index[i,j,k]]
                for l in range(0, len(tris), 3):     ## faces for this grid cell
                    edges = tris[l:l+3]
                    pts = []
                    for m in [0,1,2]:      # points in this face
                        p1 = edgeKey[edges[m]][0]
                        p2 = edgeKey[edges[m]][1]
                        v1 = data[i+p1[0], j+p1[1], k+p1[2]]
                        v2 = data[i+p2[0], j+p2[1], k+p2[2]]
                        f = (level-v1) / (v2-v1)
                        fi = 1.0 - f
                        p = (    ## interpolate between corners
                            p1[0]*fi + p2[0]*f + i + 0.5, 
                            p1[1]*fi + p2[1]*f + j + 0.5, 
                            p1[2]*fi + p2[2]*f + k + 0.5
                        ) 
                        pts.append(p)
                    facets.append(pts)

    return facets


    
def invertQTransform(tr):
    """Return a QTransform that is the inverse of *tr*.
    Rasises an exception if tr is not invertible.
    
    Note that this function is preferred over QTransform.inverted() due to
    bugs in that method. (specifically, Qt has floating-point precision issues
    when determining whether a matrix is invertible)
    """
    if not USE_WEAVE:
        raise Exception("This function depends on scipy.weave library, but it does not appear to be usable.")
    
    #return tr.inverted()[0]
    arr = np.array([[tr.m11(), tr.m12(), tr.m13()], [tr.m21(), tr.m22(), tr.m23()], [tr.m31(), tr.m32(), tr.m33()]])
    inv = scipy.linalg.inv(arr)
    return QtGui.QTransform(inv[0,0], inv[0,1], inv[0,2], inv[1,0], inv[1,1], inv[1,2], inv[2,0], inv[2,1])
    
    
