from . import Qt
from .Qt import QtCore, QtGui, QtWidgets

import numpy as np

import warnings

# from . import functions as fn # namedColorManager
from . import colormap

__all__ = ['Palette']

LEGACY_COLORS = {
    'b': QtGui.QColor(  0,  0,255,255),
    'g': QtGui.QColor(  0,255,  0,255),
    'r': QtGui.QColor(255,  0,  0,255),
    'c': QtGui.QColor(  0,255,255,255),
    'm': QtGui.QColor(255,  0,255,255),
    'y': QtGui.QColor(255,255,  0,255),
    'k': QtGui.QColor(  0,  0,  0,255),
    'w': QtGui.QColor(255,255,255,255),
    's': QtGui.QColor(100,100,150,255),
    'gr_acc':QtGui.QColor(200,200,100,255), # graphical accent color: pastel yellow
    'gr_reg':QtGui.QColor(  0,  0,255, 50)  # graphical region marker: translucent blue
}

PALETTE_DEFINITIONS = {
    'legacy': {
        'plotColorMap': {'hslCycle': {'hue':0.0}, 'sampleStep': 1/9 },
        'monoColorMap': {'mono':'neutral', 'sampleStep': 1/(9-1) },
        # --- monochrome ramp ---
        # 'm0':'#000000', 'm1':'#1e1e1e',
        # 'm2':'#353535', 'm3':'#4e4e4e',
        # 'm4':'#696969', 'm5':'#858585',
        # 'm6':'#a2a2a2', 'm7':'#c0c0c0',
        # 'm8':'#dfdfdf', 'm9':'#ffffff',
        # --- legacy colors ---
        'b': (  0,  0,255,255), 'g': (  0,255,  0,255), 'r': (255,  0,  0,255), 
        'c': (  0,255,255,255), 'm': (255,  0,255,255), 'y': (255,255,  0,255),
        'k': (  0,  0,  0,255), 'w': (255,255,255,255),
        'd': (150,150,150,255), 'l': (200,200,200,255), 's': (100,100,150,255),
        # --- manually assigned plot colors ---
        # 'p0':'l', 'p1':'y', 'p2':'r', 'p3':'m',
        # 'p4':'b', 'p5':'c', 'p6':'g', 'p7':'d',
        # --- functional colors ---
        'gr_fg' : 'd', 'gr_bg' : 'k',
        'gr_txt': 'd', 'gr_acc': (200,200,100,255),
        'gr_hlt': 'r', 'gr_reg': (  0,  0,255,100)
    },
    'relaxed': {
        'plotColorMap': {'get':'PAL-relaxed_bright', 'sampleStep':1/9 },
        'monoColorMap': {'mono':'warm', 'sampleStep':1/(9-1) },
        # 'colormap_sampling': ('CET-C6', 0.450, -0.125),
        # 'colormap_sampling': ('PAL-relaxed_bright', 0.0, 1.0/9),
        # --- extra warm (CIElab A=3 B=3) monochrome ramp ---
        # 'm0':'#1a120e', 'm1':'#2e2624',
        # 'm2':'#443c39', 'm3':'#5c5350',
        # 'm4':'#746b68', 'm5':'#8e8481',
        # 'm6':'#a89e9b', 'm7':'#c4b9b6',
        # 'm8':'#dfd5d2', 'm9':'#fcf1ee',
        # --- functional colors ---
        'gr_fg':'m5', 'gr_bg':'m0', 'gr_txt':'m7',
        'gr_acc':'#ffa84c', 'gr_hlt':'#4cb2ff', 'gr_reg': ('#b36b1e',0.63),
        # --- legacy colors ---
        'b':'p6', 'c':'p0', 'g':'p1', 'y':'p2', 'r':'p4' ,'m':'p5',
        'k':'m0', 'd':'m3', 'l':'m6', 'w': 'm9',
        's': 'gr_hlt'
    },
    'relaxed_light':{
        'plotColorMap': {'get':'PAL-relaxed', 'sampleStep': 1/9 },
        'monoColorMap': {'mono':'warm', 'sampleStep':1/(9-1) },
        # 'colormap_sampling': ('CET-C1', 0.640, -0.125),
        # 'colormap_sampling': ('PAL-relaxed', 0.0, 0.125),
        # --- slightly warm (CIElab A=1 B=2) monochrome ramp ---
        # 'm0':'#100c08', 'm1':'#262321',
        # 'm2':'#3d3a37', 'm3':'#55524f',
        # 'm4':'#6f6b68', 'm5':'#8a8683',
        # 'm6':'#a5a19e', 'm7':'#c2bdba',
        # 'm8':'#dfdbd8', 'm9':'#fdf8f5',
        # --- functional colors ---
        'gr_fg' : 'm5', 'gr_bg' : 'm9', #'#101518',
        'gr_txt': 'm2', 'gr_acc': '#ad5a00', 
        'gr_hlt': '#0080ff', 'gr_reg': ('#b36b1e',0.63),
        # legacy colors:
        'b':'p7', 'c':'p0', 'g':'#509d46', 'y':'p1', 'r':'p3' ,'m':'p5',
        'k':'m0', 'd':'m3', 'l':'m6', 'w': 'm8', 
        's': 'gr_hlt'
    },
    'pastels':{
        'plotColorMap': {'get':'CET-C7', 'subset':(0.06, 1.00), 'sampleStep':1/9 },
        'monoColorMap': {'mono':'warm', 'sampleStep':1/(9-1) },
        # 'colormap_sampling': ('CET-C7', 0.060, +0.125),
        # --- slightly warm (CIElab A=1 B=2) monochrome ramp ---
        # 'm0':'#100c08', 'm1':'#262321',
        # 'm2':'#3d3a37', 'm3':'#55524f',
        # 'm4':'#6f6b68', 'm5':'#8a8683',
        # 'm6':'#a5a19e', 'm7':'#c2bdba',
        # 'm8':'#dfdbd8', 'm9':'#fdf8f5',
        # --- functional colors ---
        'gr_fg' : 'm6', 'gr_bg' : 'm9', #'#101518',
        'gr_txt': 'm3', 'gr_acc': '#e07050', 
        'gr_hlt': '#e03020', 'gr_reg': ('#ffc0b0',0.63),
        # legacy colors:
        'b':'p3', 'c':'p4', 'g':'p5', 'y':'p6', 'r':'p1' ,'m':'p2',
        'k':'m0', 'd':'m3', 'l':'m6', 'w': 'm8', 
        's': 'gr_hlt'
    },
    'retrowave':{
        'plotColorMap': {'get':'CET-L8', 'subset':(0.275, 0.700), 'sampleStep':1/(9-1) },
        'monoColorMap': {'mono':'warm', 'sampleStep':1/(9-1) },
        # 'colormap_sampling': ('CET-L8', 0.275, 0.100),
        # --- cool monochrome ramp with no true black ---
        # 'm0':'#16212a', 'm1':'#27353e',
        # 'm2':'#3c4a54', 'm3':'#51606a',
        # 'm4':'#687782', 'm5':'#808f9a',
        # 'm6':'#98a8b3', 'm7':'#b1c1cd',
        # 'm8':'#cbdce7', 'm9':'#e6f2ff',
        # --- functional colors ---
        'gr_fg' : '#599FA6', 'gr_bg' : 'm0',
        'gr_txt': '#00E0FF', 'gr_acc': '#40B0BF',
        'gr_hlt': '#00E0FF', 'gr_reg': ('#599CA6',0.67),
        # legacy colors:
        'b':'#398bfc', 'c':'gr_txt', 'g':'#39fc49', 'y':'p7', 'r':'p4' , 'm':'p1',
        'k':'m0' , 'd': 'm3', 'l': 'm6', 'w':'m9', 's': 'gr_hlt'
    }
}

class Palette(object):
    """
    A Palette object provides a set of colors that can conveniently applied 
    to the PyQtGraph color scheme.
    It specifies at least the following colors, but additional ones can be added:
    Plot colors:
      'p0' to 'p8'  typically sampled from a color map
    Primary colors:
      'b', 'c', 'g', 'y', 'r', 'm'
    Monochrome colors:
      'm0' to 'm8'  ranging from foreground to background.
      'k', 'd', 'l', 'w'  black, dark gray, light gray and white, typically parts of the 'm0' to 'm8' range
      's' slate gray
    System colors:
      'gr_bg', 'gr_fg', 'gr_txt'  graph background, foreground and text colors
      'gr_wdw'  window background color
      'gr_reg'  partially transparent region shading color
      'gr_acc'  accent for UI elements
      'gr_hlt'  highlight for selected elements
    """
    # def __init__(self, plotColorMap=None, monoColorMap=None, colors=None ):
    
    def __init__(self, identifier, *args):
        """
        Initializes a predefined palette.
        The following identifiers are supported:
        
        =============== =========================================================================
        'legacy'        (default) Reproduces the previous PyQtGraph color scheme.
        'relaxed'       An alternative dark color scheme that avoids intense cyan and magenta.
        'relaxed_light' A variation of the 'relaxed' color scheme with a light background.
        'retrowave'     A neon/sunset color scheme inspired by retro-styled computer graphics.
        'monochrome'    A monochrome color scheme that mimics old computer screens.
                        The color can be set by an optional parameter accepted by
                        `ColorMap.makeMonochrome`. Default is 'green'.
        =============== =========================================================================
        """
        # 'application'   Attempts to extract colors from the Qt palette to match the system style.
        #                 Note that this palette is often incomplete or overriden by stylesheets.
        super().__init__()
        
        self._colors = LEGACY_COLORS.copy()
        self._isDarkMode = None # is set when assigning gr_bg color
        self.map_p = None # color map for plot color sampling
        self.map_m = None # color map for monochrome color sampling
        
        identifier = identifier.lower()
        if identifier not in PALETTE_DEFINITIONS:
            raise ValueError(f"Unknown palette definition '{identifier}''.")
        definition = PALETTE_DEFINITIONS[identifier]
        for key, prefix in ( ('plotColorMap','p'), ('monoColorMap','m') ):
            if key not in definition:
                raise KeyError(f"Palette definition does not include key '{key}'.")
            map_definition = definition.pop(key)
            cmap = None
            if 'get' in map_definition:
                cmap = colormap.get( map_definition['get'] ) # load color map
            elif 'mono' in map_definition:
                cmap = colormap.makeMonochrome( map_definition['mono'] ) # pass 'color' parameter
            elif 'hslCycle' in map_definition:
                cmap = colormap.makeHslCycle( **map_definition['hslCycle'] ) # pass keyword dictionary
            if cmap is None:
                raise ValueError(f"Invalid color map descriptor: {map_definition}")
            if 'subset' in map_definition:
                cmap = cmap.getSubset( *map_definition['subset'] )
            if prefix == 'p':
                cmap.setMappingMode('repeat')
                self.map_p = cmap
            else:
                cmap.setMappingMode('clip')
                self.map_m = cmap
            nColors = 9
            sampleStep = map_definition.get('sampleStep', 1/(nColors-1) ) # default to sample including 0.0 and 1.0
            if sampleStep is not None:
                self.sampleColorMap(colorMap=cmap, prefix=prefix, start=0., step=None, nColors=nColors)
        for key, value in definition.items():            
            print('defining:', key, value)
            alpha = None
            # unwrap (value, alpha) tuple:
            if not isinstance(value, str) and hasattr(value, '__len__') and len(value) == 2:
                value, alpha = value
                
            if value in self._colors:
                qcol = QtGui.QColor( self._colors[value] )
                if alpha is not None: qcol.setAlphaF(alpha)
                self._colors[key] = QtGui.QColor( self._colors[value] )
                continue
            else:
                print('calling QColor with', value )
                if isinstance( value, str ):
                    qcol = QtGui.QColor( value )
                    if alpha is not None: qcol.setAlphaF(alpha)
                    self._colors[key] = qcol
                    continue
                if isinstance( value, tuple ):
                    qcol = QtGui.QColor( *value )
                    if alpha is not None: qcol.setAlphaF(alpha)
                    self._colors[key] = qcol
                    continue
                else:
                    raise ValueError("Palette color specifiers must be '#[AA]RRGGBB' or (R,G,B,A) integer tuples. Found {value}.")
 
    def __getitem__(self, identifier):
        """ 
        Convenient shorthand access to palette colors.
        """
        if isinstance(identifier, str):
            return self._colors[identifier]
        if isinstance(identifier, float):
            return self.map_m[ identifier ]
        if isinstance(identifier, (int, np.integer)):
            return self.map_p[ identifier/9 ]
        if hasattr(identifier, '__len__'):
            if len(identifier) == 2:
                idx, num = identifier
                return self.map_p[ idx/num ]
        raise ValueError(f"Invalid palette color identifier {identifier}")
            
        

        # if isinstance(key, str): # access by color name
        #     return self.palette.get(key,None)
        # if isinstance(key, int): # access by plot color index
        #     idx = key % 9 # map to 0 to 8
        #     key = 'p'+str(idx)
        #     return self.palette.get(key,None)
        # return None

    # def __setitem__(self, key, color):
    #     """
    #     Convenient shorthand to add or update palette colors
    #     """
    #     if not isinstance(color, QtGui.QColor):
    #         color = QtGui.QColor(color)
    #         self._colors[key] = color 


            
    #     if isinstance(identifier, (QtGui.QColor, QtCore.Qt.GlobalColor)):
    #         return QtGui.QColor(identifier)
    #     alpha = None
        
    #     if isinstance(identifier, str):
    #         if len(identifier) <= 0: raise ValueError('Color name cannot be empty.')
    #         # if identifier[0]
                
    #     alpha = None
    #     if isinstance(identifier, str): # return known QColor
    #         name = identifier
    #         if color_dict is None or name not in color_dict:
    #             if name[0] != '#':
    #                 raise ValueError('Undefined color name '+str(identifier))
    #             return QtGui.QColor( name )
    #         else:
    #             return color_dict[name] 
    #     if not hasattr(identifier, '__len__'):
    #         raise ValueError('Invalid color definition '+str(identifier))
    #     qcol = None
    #     if len(identifier) == 2:
    #         name, alpha = identifier
    #         if color_dict is None or name not in color_dict:
    #             if name[0] != '#':
    #                 raise ValueError('Undefined color identifier '+str(identifier))
    #             qcol = QtGui.QColor( name )
    #         else:
    #             qcol = color_dict[ name ]
    #     elif len(identifier) in (3,4):
    #         qcol = QtGui.QColor( *identifier )
        
    #     if alpha is not None and qcol is not None:
    #         # distinct QColors are now created for each color
    #         # qcol = QtGui.QColor(qcol) # make a copy before changing alpha
    #         qcol.setAlpha( alpha )
    #     return qcol
        
    def setPlotColorMap(self, colorMap):
        """  Sets the color map used to sample plot colors """
        self.self.map_p = colorMap

    def plotColorMap(self):
        """ Returns the ColorMap object used to create plot colors or 'None' if not assigned. """
        return self.map_p

    def setMonoColorMap(self, colorMap):
        """  Sets the color map used to sample plot colors """
        self.self.map_m = colorMap

    def monoColorMap(self):
        """ Returns the ColorMap object used to create monochrome colors or 'None' if not assigned. """
        return self.map_m

    def sampleColorMap(self, colorMap=None, prefix='p', start=0., step=None, nColors=9  ):
        """
        Sample a ColorMap to update defined plot colors.
        When used to set plot colors (prefixed 'p') or monochrome colors (prefixed 'm'), the applied 
        color map is stored for sampling additional color. It can be retrieved with `plotColorMap()`
        or `monoColorMap()`
        =============  ===============================================================================
        **Arguments**
        cmap           (ColorMap or string)
                       A ColorMap object to be sampled or a name understood by ``colormp.get()``
                       If used to update the plot (prefix 'p') or monochrome (prefix 'm') colors, 
                       this color map will be stored to sample additional colors if needed.
                       'None' will sample a previously assigned color map if present.
        prefix         (string) 
                       default 'p' assigns colors as 'p0' to 'pXX', at least 'p8' is required.
                       Additional sets can be defined with a different prefix, e.g. 'col_0' to 'col_7'
                       All prefixes need to start with 'p', 'm' or 'col' to avoid namespace overlap with 
                       functional colors and hexadecimal numbers. 
        start          (float) 
                       first sampled value (default is 0.000)
        step           (float)
                       step between samples. Default 'None' equally samples n colors from a 
                       linear colormap, including values 0.0 and 1.0.
                       Color values > 1. and < 0. wrap around!
        nColors        default '9': Number of assigned colors. 
                       Any palette needs to include 'p0' to 'p8'.
        =============  ===============================================================================
        """
        if not (
            prefix[0]=='p' or
            prefix[0]=='m' or
            ( len(prefix)>=3 and prefix[:3]=='col')
        ):
            raise ValueError("'prefix' of plot color needs to start with 'p', 'm' or 'col'.")
        cmap = colorMap
        if cmap is None: 
            if prefix[0] != 'm': cmap = self.map_p
            else               : cmap = self.map_m
        elif isinstance(cmap, str):
            cmap = colormap.get(cmap) # obtain ColorMap if identifier is given
        if cmap is None:
            raise ValueError("Please specify 'colorMap' parameter unless a default colormap is already defined.")
        if not isinstance( cmap, colormap.ColorMap ):
            raise ValueError(f"Failed to obtain ColorMap object for 'colorMap = {cmap}'.")
            
        if prefix == 'p':
            self.map_p = cmap # replace plot color map
        elif prefix == 'm':
            self.map_n = cmap # replace monochrome color map
        if step is None:
            step = 1 / nColors # equally sample [0 to 1[ by default, appropriate for a cyclical map
        for cnt in range(nColors):
            val = start + cnt * step
            # print( val )
            if val > 1.0 or val < 0.0: # don't touch 1.0 value,
                val = val % 1. # but otherwise map to [0 to 1[ range
            qcol = cmap.mapToQColor(val)
            key = prefix + str(cnt)
            print(key, qcol.name() )
            self._colors[key] = qcol
            
    def add(self, colors):
        """
        Add colors given in dictionary 'colors' to the palette
        All colors will be converted to QColor.
        Setting 'gr_bg' with a mean color value < 127 will set the palette's 'dark' property
        """
        for key in colors:
            col = identifier_to_QColor( colors[key], color_dict=self.palette )
            if key == 'gr_bg':
                bg_tuple = col.getRgb()
                self.dark = bool( sum( bg_tuple[:3] ) < 3 * 127 ) # dark mode?
            self.palette[key] = col
            
    # def addDefaultRamp(self):
    #     """
    #     Adds a default ramp of grays m0 to m9,
    #     linearized according to CIElab lightness value
    #     """
    #     pass
    #     # todo: define based on start, intermediate, end colors
    #     # to give grays with different warmth and range of contrast

    def emulateSystem(self):
        """
        Retrieves the current Qt 'active' palette and extracts the following colors:
        =====================================  ============================================================================
        'gr_fg','gr_txt' from ColorRole.Text   (foreground color used with Base)
        'gr_bg'  from ColorRole.Base           (background color for e.g. text entry widgets)
        'gr_wdw' from ColorRole.Window         (a general background color)
        'gr_reg' from ColorRole.AlternateBase  (alternating row background color)
        'gr_acc' from ColorRole.Link           (color used for unvisited hyperlinks)
        'gr_hlt' from ColorRole.Highlight      (color to indicate a selected item)
        =====================================  ============================================================================
        """
        app = QtWidgets.QApplication.instance()
        if app is None: return None        
        qPalette = app.palette()
        col_grp = QtGui.QPalette.ColorGroup.Active
        colors = {}
        for key, alpha, col_role in (
            ('gr_bg' , None, QtGui.QPalette.ColorRole.Base),       # background color for e.g. text entry
            ('gr_fg' , None, QtGui.QPalette.ColorRole.WindowText), # overall foreground text color
            ('gr_txt', None, QtGui.QPalette.ColorRole.Text),       # foreground color used with Base
            ('gr_reg',  128, QtGui.QPalette.ColorRole.AlternateBase), # alternating row background color
            ('gr_acc', None, QtGui.QPalette.ColorRole.Link),       # color of unvisited hyperlink
            # ('gr_reg',  128, QtGui.QPalette.ColorRole.Highlight), # alternating row background color
            # ('gr_acc', None, QtGui.QPalette.ColorRole.Highlight),       # color of unvisited hyperlink
            ('gr_hlt', None, QtGui.QPalette.ColorRole.Highlight),  # color to indicate a selected item
            ('ui_wind', None, QtGui.QPalette.ColorRole.Window),    # a general background color
            ('ui_text', None, QtGui.QPalette.ColorRole.WindowText) #  overall foreground text color
        ):
            qcol = qPalette.color(col_grp, col_role)
            if alpha is not None: qcol.setAlpha(alpha)
            colors[key] = qcol
        self.add(colors)        
        colors = {
            'b': QtCore.Qt.GlobalColor.blue , 'c': QtCore.Qt.GlobalColor.cyan, 
            'g': QtCore.Qt.GlobalColor.green, 'y': QtCore.Qt.GlobalColor.yellow,
            'r': QtCore.Qt.GlobalColor.red  , 'm': QtCore.Qt.GlobalColor.magenta,
            'w': QtCore.Qt.GlobalColor.white, 's': QtCore.Qt.GlobalColor.gray,
            'k': QtCore.Qt.GlobalColor.black, 'l': QtCore.Qt.GlobalColor.lightGray,
            'd': QtCore.Qt.GlobalColor.darkGray
        }
        if not self.dark: # darken some colors for light mode
            colors['c'] = QtCore.Qt.GlobalColor.darkCyan
            colors['g'] = QtCore.Qt.GlobalColor.darkGreen
            colors['y'] = QtCore.Qt.GlobalColor.darkYellow
            colors['m'] = QtCore.Qt.GlobalColor.darkMagenta
        self.add(colors)
        colors = { 
            'p'+str(idx) : name 
            for idx, name in enumerate( ('gr_fg', 'gr_hlt', 'y','g','c','b','m','r','s') ) 
        }
        self.add(colors)

        if self.dark:
            self.map_p = colormap.get('PAL-relaxed_bright')
            cmap_m = colormap.makeMonochrome() # neutral gray ramp
            cmap_m.reverse() # switch to light-to-dark
        else:
            self.map_p = colormap.get('PAL-relaxed')
            cmap_m = colormap.makeMonochrome() # neutral gray ramp
        print('monochrome color map:', cmap_m)
        self.sampleColorMap(colorMap=cmap_m, prefix='m')  # also sets self.map_m


    def setMonochrome(self, color='green'):
        """ 
        Updates graph colors with a set based on 'monochrome' color map,
        imitating a monochrome computer screen
        ==============  =================================================================
        **Arguments:**
        color           Primary color description. Can be one of predefined identifiers
                        'green', 'amber', 'blue'
                        or a tuple of relative R,G,B contributions in range 0.0 to 1.0
        ==============  =================================================================
    """
        print('preparing monochrome map for',str(color))
        cmap = colormap.makeMonochrome(color)
        if cmap is None: 
            raise ValueError("Failed to generate color for '"+str(color)+"'")
        # define colors 'p0' (near-white) to 'p8' (dark, but visible against background)
        self.sampleColorMap( colorMap=cmap, prefix='p', start=1.0, step=-1/8 )
        # define colors 'm0' (near-white) to 'm8' (near-black):
        self.sampleColorMap( colorMap=cmap, prefix='m', start=1.0, step=-1/9) 
        colors =  {
            'gr_bg' : 'm0', 'gr_fg' : 'm4',
            'gr_txt': 'm6', 'gr_acc': 'm6',
            'gr_hlt': 'm8', 'gr_reg': ('m1',192),
            'k': 'm0', 'd': 'm1', 's': 'm3', 'l': 'm6', 'w': 'm9'
        }        
        self.add( colors )        
        # make a dictionary of plot colors (except darkest and lightest) to emulate primary colors:
        needed = { # int RGB colors that we are looking to emulate:
            'b': np.array((  0,  0,255)), 'c': np.array((  0,160,160)), 
            'g': np.array((  0,255,  0)), 'y': np.array((160,160,  0)),
            'r': np.array((255,  0,  0)), 'm': np.array((160,  0,160))
        }
        
        # project legacy colors onto monochrome reference to assigne them somewhat logically
        ref_color = np.array( self.palette['m5'].getRgb()[:3] )
        brightness = [ # estimate brightness of wanted colors "projected" into monochrome color space
            (np.sum( ref_color * needed[key] ), key) 
            for key in needed
        ]
        brightness = sorted(brightness, key=lambda brightness: brightness[0])
        # print( brightness )
        choice = ('m3','m4','m5','m6','m7','m8') # 6 candidates for 6 needed colors
        colors = {
            key: choice[idx] # assign in order of dark to bright
            for idx, (_, key) in enumerate(brightness)
        }
        self.add( colors )
        
    def apply(self):
        """
        Applies this palette to the overall PyQtGraph color scheme.
        This provides the palette to NamedColorManager, which triggers a global refresh of named colors
        """
        fn.STYLE_REGISTRY.redefinePalette( colors=self.palette )
