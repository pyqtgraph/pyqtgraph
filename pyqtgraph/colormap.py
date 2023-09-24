from collections.abc import Callable, Sequence
from os import listdir, path

import numpy as np

from .functions import clip_array, clip_scalar, colorDistance, eq, mkColor
from .Qt import QtCore, QtGui

__all__ = ['ColorMap']

_mapCache = {}

def listMaps(source=None):
    """
    .. warning:: Experimental, subject to change.

    List available color maps.

    Parameters
    ----------
    source: str, optional
        Color map source. If omitted, locally stored maps are listed. Otherwise:

          - 'matplotlib' lists maps that can be imported from Matplotlib
          - 'colorcet' lists maps that can be imported from ColorCET

    Returns
    -------
    list of str
        Known color map names.
    """
    if source is None:
        pathname = path.join(path.dirname(__file__), 'colors','maps')
        files = listdir( pathname )
        list_of_maps = []
        for filename in files:
            if filename[-4:] == '.csv' or filename[-4:] == '.hex':
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
    .. warning:: Experimental, subject to change.

    Returns a ColorMap object from a local definition or imported from another library.
    The generated ColorMap objects are cached for fast repeated access.

    Parameters
    ----------
    name: str
        Name of color map. In addition to the included maps, this can also
        be a path to a file in the local folder. See the files in the
        ``pyqtgraph/colors/maps/`` folder for examples of the format.
    source: str, optional
        If omitted, a locally stored map is returned. Otherwise:

          - 'matplotlib' imports a map defined by Matplotlib.
          - 'colorcet' imports a map defined by ColorCET.

    skipCache: bool, optional
        If `skipCache=True`, the internal cache is skipped and a new
        ColorMap object is generated. This can load an unaltered copy
        when the previous ColorMap object has been modified.
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
        elif path.isfile( filename+'.hex' ): filename += '.hex'
    with open(filename,'r') as fh:
        idx = 0
        color_list = []
        if filename[-4:].lower() != '.hex':
            csv_mode = True
        else:
            csv_mode = False
        for line in fh:
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
                try:
                    color_tuple = tuple( bytes.fromhex( hex_str ) )
                except ValueError as e:
                    raise ValueError(f"failed to convert hexadecimal value '{hex_str}'.") from e
            color_list.append( color_tuple )
            idx += 1
        # end of line reading loop
    # end of open
    cmap = ColorMap( name=name,
        pos=np.linspace(0.0, 1.0, len(color_list)),
        color=color_list) #, names=color_names)
    if cmap is not None:
        cmap.name = name
        _mapCache[name] = cmap
    return cmap

def getFromMatplotlib(name):
    """ 
    Generates a ColorMap object from a Matplotlib definition.
    Same as ``colormap.get(name, source='matplotlib')``.
    """
    # inspired and informed by "mpl_cmaps_in_ImageItem.py", published by Sebastian Hoefer at 
    # https://github.com/honkomonk/pyqtgraph_sandbox/blob/master/mpl_cmaps_in_ImageItem.py
    try:
        import matplotlib.pyplot as mpl_plt
    except ModuleNotFoundError:
        return None
    cmap = None
    col_map = mpl_plt.get_cmap(name)
    if hasattr(col_map, '_segmentdata'): # handle LinearSegmentedColormap
        data = col_map._segmentdata
        if ('red' in data) and isinstance(data['red'], (Sequence, np.ndarray)):
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
            cmap = ColorMap(pos=col_data[:,-1], color=255*col_data[:,:3]+0.5)
        # some color maps (gnuplot in particular) are defined by RGB component functions:
        elif ('red' in data) and isinstance(data['red'], Callable):
            col_data = np.zeros((64, 4))
            col_data[:,-1] = np.linspace(0., 1., 64)
            for idx, key in enumerate(['red','green','blue']):
                col_data[:,idx] = np.clip( data[key](col_data[:,-1]), 0, 1)
            cmap = ColorMap(pos=col_data[:,-1], color=255*col_data[:,:3]+0.5)
    elif hasattr(col_map, 'colors'): # handle ListedColormap
        col_data = np.array(col_map.colors)
        cmap = ColorMap( name=name,
            pos = np.linspace(0.0, 1.0, col_data.shape[0]), color=255*col_data[:,:3]+0.5 )
    if cmap is not None:
        cmap.name = name
        _mapCache[name] = cmap
    return cmap

def getFromColorcet(name):
    """ Generates a ColorMap object from a colorcet definition. Same as ``colormap.get(name, source='colorcet')``. """
    try:
        import colorcet
    except ModuleNotFoundError:
        return None
    color_strings = colorcet.palette[name]
    color_list = []
    for hex_str in color_strings:
        if hex_str[0] != '#': continue
        if len(hex_str) != 7:
            raise ValueError(f"Invalid color string '{hex_str}' in colorcet import.")
        color_tuple = tuple( bytes.fromhex( hex_str[1:] ) )
        color_list.append( color_tuple )
    if len(color_list) == 0:
        return None
    cmap = ColorMap( name=name,
        pos=np.linspace(0.0, 1.0, len(color_list)), 
        color=color_list) #, names=color_names)
    if cmap is not None:
        cmap.name = name
        _mapCache[name] = cmap
    return cmap
    
def makeHslCycle( hue=0.0, saturation=1.0, lightness=0.5, steps=36 ):
    """
    Returns a ColorMap object that traces a circular or spiraling path around the HSL color space.

    Parameters
    ----------
    hue : float or tuple of floats
        Starting point or (start, end) for hue. Values can lie outside the [0 to 1] range 
        to realize multiple cycles. For a single value, one full hue cycle is generated.
        The default starting hue is 0.0 (red). 
    saturation : float or tuple of floats, optional
        Saturation value for the colors in the cycle, in the range of [0 to 1]. 
        If a (start, end) tuple is given, saturation gradually changes between these values.
        The default saturation is 1.0.
    lightness : float or tuple of floats, optional
        Lightness value for the colors in the cycle, in the range of [0 to 1]. 
        If a (start, end) tuple is given, lightness gradually changes between these values.
        The default lightness is 1.0.
    steps: int, optional
        Number of steps in the cycle. Between these steps, the color map will interpolate in RGB space.
        The default number of steps is 36, generating a color map with 37 stops.
    """
    if isinstance( hue, (tuple, list) ):
        hueA, hueB = hue
    else:
        hueA = hue
        hueB = hueA + 1.0
    if isinstance( saturation, (tuple, list) ):
        satA, satB = saturation
    else:
        satA = satB = saturation
    if isinstance( lightness, (tuple, list) ):
        lgtA, lgtB = lightness
    else:
        lgtA = lgtB = lightness
    hue_vals = np.linspace(hueA, hueB, num=steps+1)
    sat_vals = np.linspace(satA, satB, num=steps+1)
    lgt_vals = np.linspace(lgtA, lgtB, num=steps+1)
    color_list = []
    for hue, sat, lgt in zip( hue_vals, sat_vals, lgt_vals):
        qcol = QtGui.QColor.fromHslF( hue%1.0, sat, lgt )
        color_list.append( qcol )
    name = f'Hue {hueA:0.2f}-{hueB:0.2f}'
    return ColorMap( None, color_list, name=name )

def makeMonochrome(color='neutral'):
    """
    Returns a ColorMap object with a dark to bright ramp and adjustable tint.
    
    In addition to neutral, warm or cold grays, imitations of monochrome computer monitors are also
    available. The following predefined color ramps are available:
    `neutral`, `warm`, `cool`, `green`, `amber`, `blue`, `red`, `pink`, `lavender`.
    
    The ramp can also be specified by a tuple of float values in the range of 0 to 1.
    In this case `(h, s, l0, l1)` describe hue, saturation, minimum lightness and maximum lightness
    within the HSL color space. The values `l0` and `l1` can be omitted. They default to 
    `l0=0.0` and `l1=1.0` in this case.

    Parameters
    ----------
    color: str or tuple of floats
        Color description. Can be one of the predefined identifiers, or a tuple
        `(h, s, l0, l1)`, `(h, s)` or (`h`).
        'green', 'amber', 'blue', 'red', 'lavender', 'pink'
        or a tuple of relative ``(R,G,B)`` contributions in range 0.0 to 1.0
    """
    name=f'Monochrome {color}'
    defaults = {
        'neutral': (0.00, 0.00, 0.00, 1.00),
        'warm'   : (0.10, 0.08, 0.00, 0.95),
        'cool'   : (0.60, 0.08, 0.00, 0.95),
        'green'  : (0.35, 0.55, 0.02, 0.90),
        'amber'  : (0.09, 0.80, 0.02, 0.80),
        'blue'   : (0.58, 0.85, 0.02, 0.95),
        'red'    : (0.01, 0.60, 0.02, 0.90),
        'pink'   : (0.93, 0.65, 0.02, 0.95),
        'lavender': (0.75, 0.50, 0.02, 0.90)
    }
    if isinstance(color, str):
        if color in defaults:
            h_val, s_val, l_min, l_max = defaults[color]
        else:
            valid = ','.join(defaults.keys())
            raise ValueError(f"Undefined color descriptor '{color}', known values are:\n{valid}")
    else:
        s_val = 0.70 # set up default values
        l_min = 0.00
        l_max = 1.00
        if not hasattr(color,'__len__'):
            h_val = float(color)
        elif len(color) == 1:
            h_val = color[0]
        elif len(color) == 2:
            h_val, s_val = color
        elif len(color) == 4:
            h_val, s_val, l_min, l_max = color
        else:
            raise ValueError(f"Invalid color descriptor '{color}'")
    l_vals = np.linspace(l_min, l_max, num=16)
    color_list = []
    for l_val in l_vals:
        qcol = QtGui.QColor.fromHslF( h_val, s_val, l_val )
        color_list.append( qcol )
    return ColorMap( None, color_list, name=name, linearize=True )

def modulatedBarData(length=768, width=32):
    """ 
    Returns an NumPy array that represents a modulated color bar ranging from 0 to 1.
    This is used to judge the perceived variation of the color gradient.
    
    Parameters
    ----------
    length: int
        Length of the data set. Values will vary from 0 to 1 over this axis.
    width: int
        Width of the data set. The modulation will vary from 0% to 4% over this axis.    
    """
    gradient   = np.linspace(0.00, 1.00, length)
    modulation = -0.04 * np.sin( (np.pi/4) * np.arange(length) )
    data = np.zeros( (length, width) )
    for idx in range(width):
        data[:,idx] = gradient + (idx/(width-1)) * modulation
    clip_array(data, 0.0, 1.0, out=data)
    return data

class ColorMap(object):
    """
    ColorMap(pos, color, mapping=ColorMap.CLIP)

    ColorMap stores a mapping of specific data values to colors, for example:

        | 0.0 → black
        | 0.2 → red
        | 0.6 → yellow
        | 1.0 → white

    The colors for intermediate values are determined by interpolating between
    the two nearest colors in RGB color space.

    A ColorMap object provides access to the interpolated colors by indexing with a float value:
    ``cm[0.5]`` returns a QColor corresponding to the center of ColorMap `cm`.
    """
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
        'clip': CLIP,
        'repeat': REPEAT,
        'mirror': MIRROR,
        'diverging': DIVERGING,
        'byte': BYTE,
        'float': FLOAT,
        'qcolor': QCOLOR,
    }

    def __init__(self, pos, color, mapping=CLIP, mode=None, linearize=False, name=''):
        """
        __init__(pos, color, mapping=ColorMap.CLIP)
        
        Parameters
        ----------
        pos: array_like of float, optional
            Assigned positions of specified colors. `None` sets equal spacing.
            Values need to be in range 0.0-1.0.
        color: array_like of color_like
            List of colors, interpreted via :func:`mkColor() <pyqtgraph.mkColor>`.
        mapping: str or int, optional
            Controls how values outside the 0 to 1 range are mapped to colors.
            See :func:`setMappingMode() <ColorMap.setMappingMode>` for details. 
            
            The default of `ColorMap.CLIP` continues to show
            the colors assigned to 0 and 1 for all values below or above this range, respectively.
        """
        self.name = name # storing a name helps identify ColorMaps sampled by Palette
        if pos is None:
            order = range(len(color))
            self.pos = np.linspace(0.0, 1.0, num=len(color))
        else:
            self.pos = np.array(pos)
            order = np.argsort(self.pos)
            self.pos = self.pos[order]
        
        self.color = np.zeros( (len(color), 4) ) # stores float rgba values
        for cnt, idx in enumerate(order):
            self.color[cnt] = mkColor(color[idx]).getRgbF()
        # alternative code may be more efficient, but fails to handle lists of QColor.
        # self.color = np.apply_along_axis(
        #     func1d = lambda x: np.uint8( mkColor(x).getRgb() ), # cast RGB integer values to uint8
        #     axis   = -1,
        #     arr    = color,
        #     )[order]
        
        self.mapping_mode = self.CLIP # default to CLIP mode   
        if mapping is not None:
            self.setMappingMode( mapping )
        self.stopsCache = {}
        if linearize: self.linearize()

    def setMappingMode(self, mapping):
        """
        Sets the way that values outside of the range 0 to 1 are mapped to colors.

        Parameters
        ----------
        mapping: int or str
            Sets mapping mode to

            - `ColorMap.CLIP` or 'clip': Values are clipped to the range 0 to 1. ColorMap defaults to this.
            - `ColorMap.REPEAT` or 'repeat': Colors repeat cyclically, i.e. range 1 to 2 repeats the colors for 0 to 1.
            - `ColorMap.MIRROR` or 'mirror': The range 0 to -1 uses same colors (in reverse order) as 0 to 1.
            - `ColorMap.DIVERGING` or 'diverging': Colors are mapped to -1 to 1 such that the central value appears at 0.
        """
        if isinstance(mapping, str):
            mapping = self.enumMap[mapping.lower()]
        if mapping in [self.CLIP, self.REPEAT, self.DIVERGING, self.MIRROR]:
            self.mapping_mode = mapping # only allow defined values
        else:
            raise ValueError(f"Undefined mapping type '{mapping}'")
        self.stopsCache = {}
    
    def __str__(self):
        """ provide human-readable identifier """
        if self.name is None:
            return 'unnamed ColorMap({:d})'.format(len(self.pos))
        return "ColorMap({:d}):'{:s}'".format(len(self.pos),self.name)

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

    def linearize(self):
        """
        Adjusts the positions assigned to color stops to approximately equalize the perceived color difference
        for a fixed step.
        """
        colors = self.getColors(mode=self.QCOLOR)
        distances = colorDistance(colors)
        positions = np.insert( np.cumsum(distances), 0, 0.0 )
        self.pos = positions / positions[-1] # normalize last value to 1.0
        self.stopsCache = {}

    def reverse(self):
        """
        Reverses the color map, so that the color assigned to a value of 1 now appears at 0 and vice versa.
        This is convenient to adjust imported color maps.
        """
        self.pos = 1.0 - np.flip( self.pos )
        self.color = np.flip( self.color, axis=0 )
        self.stopsCache = {}
        
    def getSubset(self, start, span):
        """
        Returns a new ColorMap object that extracts the subset specified by 'start' and 'length' 
        to the full 0.0 to 1.0 range. A negative length results in a color map that is reversed 
        relative to the original.
        
        Parameters
        ----------
        start : float
                Starting value that defines the 0.0 value of the new color map.
                Possible value between 0.0 to 1.0
        span  : float
                Span of the extracted region. The original color map will be 
                treated as cyclical if the extracted interval exceeds the 
                0.0 to 1.0 range.  Possible values between -1.0 to 1.0.
        """
        pos, col = self.getStops( mode=ColorMap.FLOAT )
        start = clip_scalar(start, 0.0, 1.0)
        span  = clip_scalar(span, -1.0, 1.0)

        if span == 0.0:
            raise ValueError("'length' needs to be non-zero")
        stop = (start + span)
        if stop > 1.0 or stop < 0.0: stop = stop % 1.0
        # find indices *inside* range, start and end will be added by sampling later
        if span > 0:
            ref_pos = start # lowest position value at start
            idxA = np.searchsorted( pos, start, side='right' )
            idxB = np.searchsorted( pos, stop , side='left'  ) # + 1 # right-side element of interval
            wraps = bool( stop < start ) # wraps around?
        else:
            ref_pos = stop # lowest position value at stop
            idxA = np.searchsorted( pos, stop , side='right')
            idxB = np.searchsorted( pos, start, side='left' ) # + 1 # right-side element of interval
            wraps = bool( stop > start ) # wraps around?
        
        if wraps: # wraps around:
            length1 = (len(pos)-idxA) # before wrap
            length2 = idxB            # after wrap
            new_length = length1 + length2 + 2 # combined; plus edge elements
            new_pos = np.zeros( new_length )
            new_col = np.zeros( (new_length, 4) )
            new_pos[ 1:length1+1] = (0 + pos[idxA:] - ref_pos) / span # starting point lie in 0 to 1 range
            new_pos[length1+1:-1] = (1 + pos[:idxB] - ref_pos) / span # end point wrapped to -1 to 0 range
            new_pos[length1] -= np.copysign(1e-6, span) # breaks degeneracy of shifted 0.0 and 1.0 values
            new_col[ 1:length1+1] = col[idxA:]
            new_col[length1+1:-1] = col[:idxB]
        else: # does not wrap around:
            new_length = (idxB - idxA) + 2 # two additional edge values will be added 
            new_pos = np.zeros( new_length )
            new_col = np.zeros( (new_length, 4) )
            new_pos[1:-1] = (pos[idxA:idxB] - ref_pos) / span
            new_col[1:-1] = col[idxA:idxB]

        if span < 0: # for reversed subsets, positions now progress 0 to -1 and need to be flipped
            new_pos += 1.0
            new_pos = np.flip( new_pos)
            new_col = np.flip( new_col, axis=0 )

        new_pos[ 0] = 0.0
        new_col[ 0] = self.mapToFloat(start)
        new_pos[-1] = 1.0
        new_col[-1] = self.mapToFloat(stop)

        cmap = ColorMap( pos=new_pos, color=255.*new_col )
        cmap.name = f"{self.name}[{start:.2f}({span:+.2f})]"
        return cmap


    def map(self, data, mode=BYTE):
        """
        map(data, mode=ColorMap.BYTE)

        Returns an array of colors corresponding to a single value or an array of values.
        Data must be either a scalar position or an array (any shape) of positions.

        Parameters
        ----------
        data: float or array_like of float
            Scalar value(s) to be mapped to colors

        mode: str or int, optional
            Determines return format:

              - `ColorMap.BYTE` or 'byte': Colors are returned as 0-255 unsigned bytes. (default)
              - `ColorMap.FLOAT` or 'float': Colors are returned as 0.0-1.0 floats.
              - `ColorMap.QCOLOR` or 'qcolor': Colors are returned as QColor objects.

        Returns
        -------
        np.ndarray of {``ColorMap.BYTE``, ``ColorMap.FLOAT``, QColor}
            for `ColorMap.BYTE` or `ColorMap.FLOAT`:

            RGB values for each `data` value, arranged in the same shape as `data`.
        list of QColor
            for `ColorMap.QCOLOR`:

            Colors for each `data` value as QColor objects.
        """
        if isinstance(mode, str):
            mode = self.enumMap[mode.lower()]

        if mode == self.QCOLOR:
            pos, color = self.getStops(self.FLOAT)
        else:
            pos, color = self.getStops(mode)

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
                return QtGui.QColor.fromRgbF(*interp)
            else:
                return [QtGui.QColor.fromRgbF(*x.tolist()) for x in interp]
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
        """Retrieve a QColor by the index of the stop it is assigned to."""
        return QtGui.QColor.fromRgbF( *self.color[idx] )

    def getGradient(self, p1=None, p2=None):
        """
        Returns a QtGui.QLinearGradient corresponding to this ColorMap.
        The span and orientation is given by two points in plot coordinates.

        When no parameters are given for `p1` and `p2`, the gradient is mapped to the
        `y` coordinates 0 to 1, unless the color map is defined for a more limited range.
        
        This is a somewhat expensive operation, and it is recommended to store and reuse the returned
        gradient instead of repeatedly regenerating it.

        Parameters
        ----------
        p1: QtCore.QPointF, optional
            Starting point (value 0) of the gradient. Default value is QPointF(0., 0.)
        p2: QtCore.QPointF, optional
            End point (value 1) of the gradient. Default parameter `dy` is the span of ``max(pos) - min(pos)``
            over which the color map is defined, typically `dy=1`.  Default is QPointF(dy, 0.)
        """
        if p1 is None:
            p1 = QtCore.QPointF(0,0)
        if p2 is None:
            p2 = QtCore.QPointF(self.pos.max()-self.pos.min(),0)
        grad = QtGui.QLinearGradient(p1, p2)

        pos, color = self.getStops(mode=self.QCOLOR)
        if self.mapping_mode == self.MIRROR:
            pos_n = (1. - np.flip(pos)) / 2
            col_n = np.flip( color, axis=0 )
            pos_p = (1. + pos) / 2
            col_p = color
            pos   = np.concatenate( (pos_n, pos_p) )
            color = np.concatenate( (col_n, col_p) )
        grad.setStops(list(zip(pos, color)))
        if self.mapping_mode == self.REPEAT:
            grad.setSpread( QtGui.QGradient.Spread.RepeatSpread )
        return grad

    def getBrush(self, span=(0.,1.), orientation='vertical'):
        """
        Returns a QBrush painting with the color map applied over the selected span of plot values.
        When the mapping mode is set to `ColorMap.MIRROR`, the selected span includes the color map twice,
        first in reversed order and then normal.
        
        It is recommended to store and reuse this gradient brush instead of regenerating it repeatedly.

        Parameters
        ----------
        span : tuple of float, optional
            Span of data values covered by the gradient:

              - Color map value 0.0 will appear at `min`,
              - Color map value 1.0 will appear at `max`.
            
            Default value is (0., 1.)

        orientation : str, default 'vertical'
            Orientation of the gradient:

              - 'vertical': `span` corresponds to the `y` coordinate.
              - 'horizontal': `span` corresponds to the `x` coordinate.
        """
        if orientation == 'vertical':
            grad = self.getGradient( p1=QtCore.QPointF(0.,span[0]), p2=QtCore.QPointF(0.,span[1]) )
        elif orientation == 'horizontal':
            grad = self.getGradient( p1=QtCore.QPointF(span[0],0.), p2=QtCore.QPointF(span[1],0.) )
        else:
            raise ValueError("Orientation must be 'vertical' or 'horizontal'")
        return QtGui.QBrush(grad)

    def getPen(self, span=(0.,1.), orientation='vertical', width=1.0):
        """
        Returns a QPen that draws according to the color map based on vertical or horizontal position.
        
        It is recommended to store and reuse this gradient pen instead of regenerating it repeatedly.


        Parameters
        ----------
        span : tuple of float
            Span of the data values covered by the gradient:

              - Color map value 0.0 will appear at `min`.
              - Color map value 1.0 will appear at `max`.

            Default is (0., 1.)
        orientation : str, default 'vertical'
            Orientation of the gradient:

              - 'vertical' creates a vertical gradient, where `span` corresponds to the `y` coordinate.
              - 'horizontal' creates a horizontal gradient, where `span` corresponds to the `x` coordinate.

        width : int or float
            Width of the pen in pixels on screen.
        """
        brush = self.getBrush( span=span, orientation=orientation )
        pen = QtGui.QPen(brush, width)
        pen.setCosmetic(True)
        return pen

    def getColors(self, mode=BYTE):
        """
        Returns a list of the colors associated with the stops of the color map.
        
        The parameter `mode` can be one of
            - `ColorMap.BYTE` or 'byte' to return colors as RGBA tuples in byte format (0 to 255)
            - `ColorMap.FLOAT` or 'float' to return colors as RGBA tuples in float format (0.0 to 1.0)
            - `ColorMap.QCOLOR` or 'qcolor' to return a list of QColors
            
        The default is byte format.
        """
        stops, color = self.getStops(mode=mode)
        return color

    def getStops(self, mode=BYTE):
        """
        Returns a tuple (stops, colors) containing a list of all stops (ranging 0.0 to 1.0)
        and a list of the associated colors.
        
        The parameter `mode` can be one of
            - `ColorMap.BYTE` or 'byte' to return colors as RGBA tuples in byte format (0 to 255)
            - `ColorMap.FLOAT` or 'float' to return colors as RGBA tuples in float format (0.0 to 1.0)
            - `ColorMap.QCOLOR` or 'qcolor' to return a list of QColors

        The default is byte format.
        """
        if isinstance(mode, str):
            mode = self.enumMap[mode.lower()]

        if mode not in self.stopsCache:
            color = self.color
            if mode == self.BYTE and color.dtype.kind == 'f':
                color = (color*255).astype(np.ubyte)
            elif mode == self.FLOAT and color.dtype.kind != 'f':
                color = color.astype(float) / 255.
            elif mode == self.QCOLOR:
                if color.dtype.kind == 'f':
                    factory = QtGui.QColor.fromRgbF
                else:
                    factory = QtGui.QColor.fromRgb
                color = [factory(*x.tolist()) for x in color]
            self.stopsCache[mode] = (self.pos, color)
        return self.stopsCache[mode]

    def getLookupTable(self, start=0.0, stop=1.0, nPts=512, alpha=None, mode=BYTE):
        """
        getLookupTable(start=0.0, stop=1.0, nPts=512, alpha=None, mode=ColorMap.BYTE)

        Returns an equally-spaced lookup table of RGB(A) values created
        by interpolating the specified color stops.

        Parameters
        ----------
        start:  float, default=0.0
            The starting value in the lookup table
        stop: float, default=1.0
            The final value in the lookup table
        nPts: int, default=512
            The number of points in the returned lookup table.
        alpha: bool, optional
            Specifies whether or not alpha values are included in the table.
            If alpha is None, it will be automatically determined.
        mode: int or str, default='byte'
            Determines return type as described in :func:`map() <pyqtgraph.ColorMap.map>`, can be
            either `ColorMap.BYTE` (0 to 255), `ColorMap.FLOAT` (0.0 to 1.0) or `ColorMap.QColor`.

        Returns
        -------
        np.ndarray of {``ColorMap.BYTE``, ``ColorMap.FLOAT``}
            for `ColorMap.BYTE` or `ColorMap.FLOAT`:

            RGB values for each `data` value, arranged in the same shape as `data`.
            If alpha values are included the array has shape (`nPts`, 4), otherwise (`nPts`, 3).
    
        list of QColor
            for `ColorMap.QCOLOR`:

            Colors for each `data` value as QColor objects.
        """
        if isinstance(mode, str):
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
        """Returns `True` if any stops have assigned colors with alpha < 255."""
        max = 1.0 if self.color.dtype.kind == 'f' else 255
        return np.any(self.color[:,3] != max)

    def isMapTrivial(self):
        """
        Returns `True` if the gradient has exactly two stops in it: Black at 0.0 and white at 1.0.
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
