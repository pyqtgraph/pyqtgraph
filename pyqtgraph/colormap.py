import numpy as np
from .Qt import QtGui, QtCore
from .python2_3 import basestring
from .functions import mkColor, eq
from os import path, listdir
from collections.abc import Callable, Sequence

_mapCache = {}

def listMaps(source=None):
    """
    Warning, highly experimental, subject to change.

    List available color maps
    ===============  =================================================================
    **Arguments:**
    source           'matplotlib' lists maps that can be imported from MatPlotLib
                     'colorcet' lists maps that can be imported from ColorCET
                     otherwise local maps are listed
    ===============  =================================================================
    """
    if source is None:
        pathname = path.join(path.dirname(__file__), 'colors','maps')
        files = listdir( pathname )
        list_of_maps = []
        for filename in files:
            if filename[-4:] == '.csv':
                list_of_maps.append(filename[:-4])
        return list_of_maps
    elif source.lower() == 'matplotlib':
        try:
            import matplotlib.pyplot as mpl_plt
            list_of_maps = mpl_plt.colormaps()
            return list_of_maps
        except ModuleNotFoundError: 
            return []
    elif source.lower() == 'colorcet':
        try:
            import colorcet
            list_of_maps = list( colorcet.palette.keys() )
            list_of_maps.sort()
            return list_of_maps
        except ModuleNotFoundError: 
            return []
    return []    


def get(name, source=None, skipCache=False):
    """
    Warning, highly experimental, subject to change.

    Returns a ColorMap object from a local definition or imported from another library
    ===============  =================================================================
    **Arguments:**
    name             Name of color map. Can be a path to a defining file.
    source           'matplotlib' imports a map defined by Matplotlib
                     'colorcet' imports a maps defined by ColorCET
                     otherwise local data is used
    ===============  =================================================================
    """
    if not skipCache and name in _mapCache:
        return _mapCache[name]
    if source is None:
        return _getFromFile(name)
    elif source == 'matplotlib':
        return getFromMatplotlib(name)
    elif source == 'colorcet':
        return getFromColorcet(name)
    return None

def _getFromFile(name):
    filename = name
    if filename[0] !='.': # load from built-in directory
        dirname = path.dirname(__file__)
        filename = path.join(dirname, 'colors/maps/'+filename)
    if not path.isfile( filename ): # try suffixes if file is not found:
        if   path.isfile( filename+'.csv' ): filename += '.csv'
        elif path.isfile( filename+'.txt' ): filename += '.txt'
    with open(filename,'r') as fh:
        idx = 0
        color_list = []
        if filename[-4:].lower() != '.txt':
            csv_mode = True
        else:
            csv_mode = False
        for line in fh:
            name = None
            line = line.strip()
            if len(line) == 0: continue # empty line
            if line[0] == ';': continue # comment
            parts = line.split(sep=';', maxsplit=1) # split into color and names/comments
            if csv_mode:
                comp = parts[0].split(',')
                if len( comp ) < 3: continue # not enough components given
                color_tuple = tuple( [ int(255*float(c)+0.5) for c in comp ] )
            else:
                hex_str = parts[0]
                if hex_str[0] == '#': 
                    hex_str = hex_str[1:] # strip leading #
                if len(hex_str) < 3: continue # not enough information
                if len(hex_str) == 3: # parse as abbreviated RGB
                    hex_str = 2*hex_str[0] + 2*hex_str[1] + 2*hex_str[2]
                elif len(hex_str) == 4: # parse as abbreviated RGBA
                    hex_str = 2*hex_str[0] + 2*hex_str[1] + 2*hex_str[2] + 2*hex_str[3]
                if len(hex_str) < 6: continue # not enough information
                color_tuple = tuple( bytes.fromhex( hex_str ) )
            color_list.append( color_tuple )
            idx += 1
        # end of line reading loop
    # end of open
    cm = ColorMap(
        pos=np.linspace(0.0, 1.0, len(color_list)), 
        color=color_list) #, names=color_names)
    _mapCache[name] = cm
    return cm

def getFromMatplotlib(name):
    """ import colormap from matplotlib definition """
    # inspired and informed by "mpl_cmaps_in_ImageItem.py", published by Sebastian Hoefer at 
    # https://github.com/honkomonk/pyqtgraph_sandbox/blob/master/mpl_cmaps_in_ImageItem.py
    try:
        import matplotlib.pyplot as mpl_plt
    except ModuleNotFoundError:
        return None
    cm = None
    col_map = mpl_plt.get_cmap(name)
    if hasattr(col_map, '_segmentdata'): # handle LinearSegmentedColormap
        data = col_map._segmentdata
        if ('red' in data) and isinstance(data['red'], Sequence):
            positions = set() # super-set of handle positions in individual channels
            for key in ['red','green','blue']:
                for tup in data[key]:
                    positions.add(tup[0])
            col_data = np.zeros((len(positions),4 ))
            col_data[:,-1] = sorted(positions)
            for idx, key in enumerate(['red','green','blue']):
                positions = np.zeros( len(data[key] ) )
                comp_vals = np.zeros( len(data[key] ) )
                for idx2, tup in enumerate( data[key] ):
                    positions[idx2] = tup[0]
                    comp_vals[idx2] = tup[1] # these are sorted in the raw data
                col_data[:,idx] = np.interp(col_data[:,3], positions, comp_vals)
            cm = ColorMap(pos=col_data[:,-1], color=255*col_data[:,:3]+0.5)
        # some color maps (gnuplot in particular) are defined by RGB component functions:
        elif ('red' in data) and isinstance(data['red'], Callable):
            col_data = np.zeros((64, 4))
            col_data[:,-1] = np.linspace(0., 1., 64)
            for idx, key in enumerate(['red','green','blue']):
                col_data[:,idx] = np.clip( data[key](col_data[:,-1]), 0, 1)
            cm = ColorMap(pos=col_data[:,-1], color=255*col_data[:,:3]+0.5)  
    elif hasattr(col_map, 'colors'): # handle ListedColormap
        col_data = np.array(col_map.colors)
        cm = ColorMap(pos=np.linspace(0.0, 1.0, col_data.shape[0]), color=255*col_data[:,:3]+0.5 )
    if cm is not None:
        _mapCache[name] = cm
    return cm

def getFromColorcet(name):
    """ import colormap from colorcet definition """
    try:
        import colorcet
    except ModuleNotFoundError:
        return None
    color_strings = colorcet.palette[name]
    color_list = []
    for hex_str in color_strings:
        if hex_str[0] != '#': continue
        if len(hex_str) != 7:            
            raise ValueError('Invalid color string '+str(hex_str)+' in colorcet import.')
        color_tuple = tuple( bytes.fromhex( hex_str[1:] ) )
        color_list.append( color_tuple )
    if len(color_list) == 0: 
        return None
    cm = ColorMap(
        pos=np.linspace(0.0, 1.0, len(color_list)), 
        color=color_list) #, names=color_names)
    _mapCache[name] = cm
    return cm


class ColorMap(object):
    """
    A ColorMap defines a relationship between a scalar value and a range of colors. 
    ColorMaps are commonly used for false-coloring monochromatic images, coloring 
    scatter-plot points, and coloring surface plots by height. 

    Each color map is defined by a set of colors, each corresponding to a
    particular scalar value. For example:

        | 0.0  -> black
        | 0.2  -> red
        | 0.6  -> yellow
        | 1.0  -> white

    The colors for intermediate values are determined by interpolating between 
    the two nearest colors in either RGB or HSV color space.

    To provide user-defined color mappings, see :class:`GradientWidget <pyqtgraph.GradientWidget>`.
    """

    ## color interpolation modes
    RGB = 1
    HSV_POS = 2
    HSV_NEG = 3

    ## mapping modes
    CLIP   = 1
    REPEAT = 2
    MIRROR = 3
    DIVERGING = 4

    ## return types
    BYTE = 1
    FLOAT = 2
    QCOLOR = 3

    enumMap = {
        'rgb': RGB,
        # 'hsv+': HSV_POS,
        # 'hsv-': HSV_NEG,
        # 'clip': CLIP,
        # 'repeat': REPEAT,
        'byte': BYTE,
        'float': FLOAT,
        'qcolor': QCOLOR,
    }

    def __init__(self, pos, color, mode=None, mapping=None): #, names=None):
        """
        ===============     =================================================================
        **Arguments:**
        pos                 Array of positions where each color is defined
        color               Array of colors.
                            Values are interpreted via 
                            :func:`mkColor() <pyqtgraph.mkColor>`.
        mode                Array of color modes (ColorMap.RGB, HSV_POS, or HSV_NEG)
                            indicating the color space that should be used when
                            interpolating between stops. Note that the last mode value is
                            ignored. By default, the mode is entirely RGB.
        mapping             Mapping mode (ColorMap.CLIP, REPEAT, MIRROR, or DIVERGING)
                            controlling mapping of relative index to color. 
                            CLIP maps colors to [0.0;1.0]
                            REPEAT maps colors to repeating intervals [0.0;1.0];[1.0-2.0],...
                            MIRROR maps colors to [0.0;-1.0] and [0.0;+1.0] identically
                            DIVERGING maps colors to [-1.0;+1.0]
        ===============     =================================================================
        """
        self.pos = np.array(pos)
        order = np.argsort(self.pos)
        self.pos = self.pos[order]
        self.color = np.apply_along_axis(
            func1d = lambda x: mkColor(x).getRgb(),
            axis   = -1,
            arr    = color,
            )[order]
        if mode is None:
            mode = np.ones(len(pos))
        self.mode = mode

        if mapping is None:
            self.mapping_mode = self.CLIP
        elif mapping == self.REPEAT:
            self.mapping_mode = self.REPEAT
        elif mapping == self.DIVERGING:
            self.mapping_mode = self.DIVERGING
        elif mapping == self.MIRROR:
            self.mapping_mode = self.MIRROR
        else:
            self.mapping_mode = self.CLIP
        
        self.stopsCache = {}

    def __getitem__(self, key):
        """ Convenient shorthand access to palette colors """
        if isinstance(key, int): # access by color index 
            return self.getByIndex(key)
        # otherwise access by map
        try: # accept any numerical format that converts to float
            float_idx = float(key) 
            return self.mapToQColor(float_idx)
        except ValueError: pass
        return None

    def map(self, data, mode='byte'):
        """
        Return an array of colors corresponding to the values in *data*. 
        Data must be either a scalar position or an array (any shape) of positions.
        
        The *mode* argument determines the type of data returned:

        =========== ===============================================================
        byte        (default) Values are returned as 0-255 unsigned bytes.
        float       Values are returned as 0.0-1.0 floats. 
        qcolor      Values are returned as an array of QColor objects.
        =========== ===============================================================
        """
        if isinstance(mode, basestring):
            mode = self.enumMap[mode.lower()]
            
        if mode == self.QCOLOR:
            pos, color = self.getStops(self.BYTE)
        else:
            pos, color = self.getStops(mode)

        # Interpolate
        # TODO: is griddata faster?
        #          interp = scipy.interpolate.griddata(pos, color, data)
        if np.isscalar(data):
            interp = np.empty((color.shape[1],), dtype=color.dtype)
        else:
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            interp = np.empty(data.shape + (color.shape[1],), dtype=color.dtype)

        if self.mapping_mode != self.CLIP:
            if self.mapping_mode == self.REPEAT:
                data = data % 1.0
            elif self.mapping_mode == self.DIVERGING:
                data = (data/2)+0.5
            elif self.mapping_mode == self.MIRROR:
                data = abs(data)

        for i in range(color.shape[1]):
            interp[...,i] = np.interp(data, pos, color[:,i])

        # Convert to QColor if requested
        if mode == self.QCOLOR:
            if np.isscalar(data):
                return QtGui.QColor(*interp)
            else:
                return [QtGui.QColor(*x) for x in interp]
        else:
            return interp
        
    def mapToQColor(self, data):
        """Convenience function; see :func:`map() <pyqtgraph.ColorMap.map>`."""
        return self.map(data, mode=self.QCOLOR)

    def mapToByte(self, data):
        """Convenience function; see :func:`map() <pyqtgraph.ColorMap.map>`."""
        return self.map(data, mode=self.BYTE)

    def mapToFloat(self, data):
        """Convenience function; see :func:`map() <pyqtgraph.ColorMap.map>`."""
        return self.map(data, mode=self.FLOAT)

    def getByIndex(self, idx):
        """Retrieve palette QColor by index"""
        return QtGui.QColor( *self.color[idx] )

    def getGradient(self, p1=None, p2=None):
        """Return a QLinearGradient object spanning from QPoints p1 to p2."""
        if p1 == None:
            p1 = QtCore.QPointF(0,0)
        if p2 == None:
            p2 = QtCore.QPointF(self.pos.max()-self.pos.min(),0)
        g = QtGui.QLinearGradient(p1, p2)
        
        pos, color = self.getStops(mode=self.BYTE)
        color = [QtGui.QColor(*x) for x in color]
        g.setStops(list(zip(pos, color)))
        return g

    def getColors(self, mode=None):
        """Return list of all color stops converted to the specified mode.
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
            self.stopsCache[mode] = (self.pos, color)
        return self.stopsCache[mode]

    def getLookupTable(self, start=0.0, stop=1.0, nPts=512, alpha=None, mode='byte'):
        """
        Return an RGB(A) lookup table (ndarray). 
        
        ===============   =============================================================================
        **Arguments:**
        start             The starting value in the lookup table (default=0.0)
        stop              The final value in the lookup table (default=1.0)
        nPts              The number of points in the returned lookup table.
        alpha             True, False, or None - Specifies whether or not alpha values are included
                          in the table. If alpha is None, it will be automatically determined.
        mode              Determines return type: 'byte' (0-255), 'float' (0.0-1.0), or 'qcolor'.
                          See :func:`map() <pyqtgraph.ColorMap.map>`.
        ===============   =============================================================================
        """
        if isinstance(mode, basestring):
            mode = self.enumMap[mode.lower()]

        if alpha is None:
            alpha = self.usesAlpha()

        x = np.linspace(start, stop, nPts)
        table = self.map(x, mode)

        if not alpha and mode != self.QCOLOR:
            return table[:,:3]
        else:
            return table

    def usesAlpha(self):
        """Return True if any stops have an alpha < 255"""
        max = 1.0 if self.color.dtype.kind == 'f' else 255
        return np.any(self.color[:,3] != max)

    def isMapTrivial(self):
        """
        Return True if the gradient has exactly two stops in it: black at 0.0 and white at 1.0.
        """
        if len(self.pos) != 2:
            return False
        if self.pos[0] != 0.0 or self.pos[1] != 1.0:
            return False
        if self.color.dtype.kind == 'f':
            return np.all(self.color == np.array([[0.,0.,0.,1.], [1.,1.,1.,1.]]))
        else:
            return np.all(self.color == np.array([[0,0,0,255], [255,255,255,255]]))

    def __repr__(self):
        pos = repr(self.pos).replace('\n', '')
        color = repr(self.color).replace('\n', '')
        return "ColorMap(%s, %s)" % (pos, color)

    def __eq__(self, other):
        if other is None:
            return False
        return eq(self.pos, other.pos) and eq(self.color, other.color)
