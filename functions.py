# -*- coding: utf-8 -*-
"""
functions.py -  Miscellaneous functions with no other home
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more infomation.
"""

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

SI_PREFIXES = u'yzafpnµm kMGTPEZY'
SI_PREFIXES_ASCII = 'yzafpnum kMGTPEZY'

USE_WEAVE = True


from Qt import QtGui, QtCore
import numpy as np
import scipy.ndimage
import decimal, re
import scipy.weave
import debug

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
        print x, type(x)
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
        plusminus = space + u"±" + space
        fmt = "%." + str(precision) + u"g%s%s%s%s"
        return fmt % (x*p, pref, suffix, plusminus, siFormat(error, precision=precision, suffix=suffix, space=space, minVal=minVal))
    
def siEval(s):
    """
    Convert a value written in SI notation to its equivalent prefixless value
    
    Example::
    
        siEval("100 μV")  # returns 0.0001
    """
    
    s = unicode(s)
    m = re.match(r'(-?((\d+(\.\d*)?)|(\.\d+))([eE]-?\d+)?)\s*([u' + SI_PREFIXES + r']?)$', s)
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
    args = map(lambda a: 0 if np.isnan(a) or np.isinf(a) else a, args)
    args = map(int, args)
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
            return arg
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

def hsvColor(h, s=1.0, v=1.0, a=1.0):
    """Generate a QColor from HSVa values."""
    c = QtGui.QColor()
    c.setHsvF(h, s, v, a)
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


def affineSlice(data, shape, origin, vectors, axes, **kargs):
    """
    Take a slice of any orientation through an array. This is useful for extracting sections of multi-dimensional arrays such as MRI images for viewing as 1D or 2D data.
    
    The slicing axes are aribtrary; they do not need to be orthogonal to the original data or even to each other. It is possible to use this function to extract arbitrary linear, rectangular, or parallelepiped shapes from within larger datasets.
    
    For a graphical interface to this function, see :func:`ROI.getArrayRegion`
    
    Arguments:
    
        | *data* (ndarray): the original dataset
        | *shape*: the shape of the slice to take (Note the return value may have more dimensions than len(shape))
        | *origin*: the location in the original dataset that will become the origin in the sliced data.
        | *vectors*: list of unit vectors which point in the direction of the slice axes
        
        * each vector must have the same length as *axes*
        * If the vectors are not unit length, the result will be scaled.
        * If the vectors are not orthogonal, the result will be sheared.
            
        *axes*: the axes in the original dataset which correspond to the slice *vectors*
        
    Example: start with a 4D fMRI data set, take a diagonal-planar slice out of the last 3 axes
        
        * data = array with dims (time, x, y, z) = (100, 40, 40, 40)
        * The plane to pull out is perpendicular to the vector (x,y,z) = (1,1,1) 
        * The origin of the slice will be at (x,y,z) = (40, 0, 0)
        * We will slice a 20x20 plane from each timepoint, giving a final shape (100, 20, 20)
        
    The call for this example would look like::
        
        affineSlice(data, shape=(20,20), origin=(40,0,0), vectors=((-1, 1, 0), (-1, 0, 1)), axes=(1,2,3))
    
    Note the following must be true: 
        
        | len(shape) == len(vectors) 
        | len(origin) == len(axes) == len(vectors[0])
    """
    
    # sanity check
    if len(shape) != len(vectors):
        raise Exception("shape and vectors must have same length.")
    if len(origin) != len(axes):
        raise Exception("origin and axes must have same length.")
    for v in vectors:
        if len(v) != len(axes):
            raise Exception("each vector must be same length as axes.")
        
    shape = map(np.ceil, shape)

    ## transpose data so slice axes come first
    trAx = range(data.ndim)
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
        output[ind] = scipy.ndimage.map_coordinates(data[ind], x, **kargs)
    
    tr = range(output.ndim)
    trb = []
    for i in range(min(axes)):
        ind = tr1.index(i) + (len(shape)-len(axes))
        tr.remove(ind)
        trb.append(ind)
    tr2 = tuple(trb+tr)

    ## Untranspose array before returning
    return output.transpose(tr2)





def makeARGB(data, lut=None, levels=None):
    """
    Convert a 2D or 3D array into an ARGB array suitable for building QImages
    Will optionally do scaling and/or table lookups to determine final colors.
    
    Returns the ARGB array and a boolean indicating whether there is alpha channel data.
    
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
                    for i in xrange(levels.shape[0]):
                        scale = float(lutLength / (levels[i,1]-levels[i,0]))
                        offset = float(levels[i,0])
                        newData[...,i] = rescaleData(data, scale, offset)
                elif data.ndim == 3:
                    newData = np.empty(data.shape, dtype=np.uint32)
                    for i in xrange(data.shape[2]):
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


    order = [2,1,0,3] ## for some reason, the colors line up as BGR in the final image.
    if data.shape[2] == 1:
        for i in xrange(3):
            imgData[..., order[i]] = data[..., 0]    
    else:
        for i in xrange(0, data.shape[2]):
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
    except AttributeError:
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
    
    