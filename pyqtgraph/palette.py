from . import Qt
from .Qt import QtCore, QtGui, QtWidgets

import numpy as np
import math

from . import functions as fn # namedColorManager
from . import colormap

__all__ = ['Palette']

#### todo list ####
# find color definitions for relaxed-light

PALETTE_DEFINITIONS = {
    'legacy': {
        'colormap_sampling' : None,
        # --- monochrome ramp ---
        'm0':'#000000', 'm1':'#1e1e1e',
        'm2':'#353535', 'm3':'#4e4e4e',
        'm4':'#696969', 'm5':'#858585',
        'm6':'#a2a2a2', 'm7':'#c0c0c0',
        'm8':'#dfdfdf', 'm9':'#ffffff',
        # --- legacy colors ---
        'b': (  0,  0,255,255), 'g': (  0,255,  0,255), 'r': (255,  0,  0,255), 
        'c': (  0,255,255,255), 'm': (255,  0,255,255), 'y': (255,255,  0,255),
        'k': (  0,  0,  0,255), 'w': (255,255,255,255),
        'd': (150,150,150,255), 'l': (200,200,200,255), 's': (100,100,150,255),
        # --- manually assigned plot colors ---
        'p0':'l', 'p1':'y', 'p2':'r', 'p3':'m',
        'p4':'b', 'p5':'c', 'p6':'g', 'p7':'d',
        # --- functional colors ---
        'gr_fg' : 'd', 'gr_bg' : 'k',
        'gr_txt': 'd', 'gr_acc': (200,200,100,255),
        'gr_hlt': 'r', 'gr_reg': (  0,  0,255,100)
    },
    'relaxed_dark':{
        'colormap_sampling': ('CET-C6', 0.450, -0.125),
        # --- extra warm (CIElab A=3 B=3) monochrome ramp ---
        'm0':'#1a120e', 'm1':'#2e2624',
        'm2':'#443c39', 'm3':'#5c5350',
        'm4':'#746b68', 'm5':'#8e8481',
        'm6':'#a89e9b', 'm7':'#c4b9b6',
        'm8':'#dfd5d2', 'm9':'#fcf1ee',
        # --- functional colors ---
        'gr_fg':'m5', 'gr_bg':'m0', 'gr_txt':'m7',
        'gr_acc':'#ffa84c', 'gr_hlt':'#4cb2ff', 'gr_reg': ('#b36b1e',160),
        # --- legacy colors ---
        'b':'p6', 'c':'p0', 'g':'p1', 'y':'p2', 'r':'p4' ,'m':'p5',
        'k':'m0', 'd':'m3', 'l':'m6', 'w': 'm9',
        's': 'gr_hlt'
    },
    'relaxed_light':{
        'colormap_sampling': ('CET-C1', 0.640, -0.125),
        # --- slightly warm (CIElab A=1 B=2) monochrome ramp ---
        'm0':'#100c08', 'm1':'#262321',
        'm2':'#3d3a37', 'm3':'#55524f',
        'm4':'#6f6b68', 'm5':'#8a8683',
        'm6':'#a5a19e', 'm7':'#c2bdba',
        'm8':'#dfdbd8', 'm9':'#fdf8f5',
        # --- functional colors ---
        'gr_fg' : 'm5', 'gr_bg' : 'm9', #'#101518',
        'gr_txt': 'm2', 'gr_acc': '#ad5a00', 
        'gr_hlt': '#0080ff', 'gr_reg': ('#b36b1e',160),
        # legacy colors:
        'b':'p7', 'c':'p0', 'g':'#509d46', 'y':'p1', 'r':'p3' ,'m':'p5',
        'k':'m0', 'd':'m3', 'l':'m6', 'w': 'm9', 
        's': 'gr_hlt'
    },
    'pastels':{
        'colormap_sampling': ('CET-C7', 0.060, +0.125),
        # --- slightly warm (CIElab A=1 B=2) monochrome ramp ---
        'm0':'#100c08', 'm1':'#262321',
        'm2':'#3d3a37', 'm3':'#55524f',
        'm4':'#6f6b68', 'm5':'#8a8683',
        'm6':'#a5a19e', 'm7':'#c2bdba',
        'm8':'#dfdbd8', 'm9':'#fdf8f5',
        # --- functional colors ---
        'gr_fg' : 'm6', 'gr_bg' : 'm9', #'#101518',
        'gr_txt': 'm3', 'gr_acc': '#e07050', 
        'gr_hlt': '#e03020', 'gr_reg': ('#ffc0b0',160),
        # legacy colors:
        'b':'p3', 'c':'p4', 'g':'p5', 'y':'p6', 'r':'p1' ,'m':'p2',
        'k':'m0', 'd':'m3', 'l':'m6', 'w': 'm9', 
        's': 'gr_hlt'
    },
    'synthwave':{
        'colormap_sampling': ('CET-L8', 0.275, 0.100),
        # --- cool monochrome ramp with no true black ---
        'm0':'#16212a', 'm1':'#27353e',
        'm2':'#3c4a54', 'm3':'#51606a',
        'm4':'#687782', 'm5':'#808f9a',
        'm6':'#98a8b3', 'm7':'#b1c1cd',
        'm8':'#cbdce7', 'm9':'#e6f2ff',
        # --- functional colors ---
        'gr_fg' : '#599FA6', 'gr_bg' : 'm0',
        'gr_txt': '#00E0FF', 'gr_acc': '#40B0BF',
        'gr_hlt': '#00E0FF', 'gr_reg': ('#599CA6',170),
        # legacy colors:
        'b':'#398bfc', 'c':'gr_txt', 'g':'#39fc49', 'y':'p7', 'r':'p4' , 'm':'p1',
        'k':'m0' , 'd': 'm3', 'l': 'm6', 'w':'m9', 's': 'gr_hlt'
    }
}

def identifier_to_QColor( identifier, color_dict=None ):
    """ 
    Convert color information to a QColor 
    ===================  =============================================================
    **allowed formats**    
    'name'               name must be a hex value or a key in 'color_dict'
    ('name', alpha)      new copy will be assigned the specified alpha value
    QColor               will be copied to avoid interaction if changing alpha values
    Qt.GlobalColor       will result in a matching QColor 
    (R,G,B)              a new Qcolor will be created
    (R,G,B, alpha)       a new Qcolor with specified opacity will be created
    ===================  =============================================================
    """
    if isinstance(identifier, (QtGui.QColor, QtCore.Qt.GlobalColor)):
        return QtGui.QColor(identifier)
    alpha = None
    if isinstance(identifier, str): # return known QColor
        name = identifier
        if color_dict is None or name not in color_dict:
            if name[0] != '#':
                raise ValueError('Undefined color name '+str(identifier))
            return QtGui.QColor( name )
        else:
            return color_dict[name] 
    if not hasattr(identifier, '__len__'):
        raise ValueError('Invalid color definition '+str(identifier))
    qcol = None
    if len(identifier) == 2:
        name, alpha = identifier
        if color_dict is None or name not in color_dict:
            if name[0] != '#':
                raise ValueError('Undefined color identifier '+str(identifier))
            qcol = QtGui.QColor( name )
        else:
            qcol = color_dict[ name ]
    elif len(identifier) in (3,4):
        qcol = QtGui.QColor( *identifier )
    
    if alpha is not None and qcol is not None:
        # distinct QColors are now created for each color
        # qcol = QtGui.QColor(qcol) # make a copy before changing alpha
        qcol.setAlpha( alpha )
    return qcol


def get(identifier, *args):
    """
    Returns a Palette object that can be applied to update the PyQtGraph color scheme
    ===============  ====================================================================
     **Arguments:**
    identifier       'system' (default): Colors are based on current Qt QPalette
                     'legacy': The color scheme of previous versions of PyQtGraph
                     'monochrome' ['color identifier']: Creates a palette that imitates
                        a monochrome computer monitor.
                        'color identifier' can be one of 
                        'green', 'amber', 'blue', 'red', 'lavender', 'pink'
                        or a tuple of relative R,G,B contributions in range 0.0 to 1.0

                     {dictionary}: full palette specification, see palette.py for details
    ===============  ====================================================================
    """    
    if identifier == 'system':
        pal = Palette()
        return pal # default QPalette based settings
    if identifier == 'monochrome':
        pal = Palette()
        pal.setMonochrome( *args )
        return pal
    if identifier in PALETTE_DEFINITIONS:
        info = PALETTE_DEFINITIONS[identifier].copy()
        sampling_info = info.pop('colormap_sampling', None)
        pal = Palette( cmap=sampling_info, colors=info )
        return pal
    raise KeyError("Unknown palette identifier '"+str(identifier)+"'")

class Palette(object):
    """
    A Palette object provides a set of colors that can conveniently applied 
    to the PyQtGraph color scheme.
    It specifies at least the following colors, but additional one can be added:
    Primary colors:
      'b', 'c', 'g', 'y', 'r', 'm'
    Gray scale:
      'k', 'd', 'l', 'w'  ranging from black to white
      's' slate gray
    System colors:
      'gr_bg', 'gr_fg', 'gr_txt'  graph background, foreground and text colors
      'gr_wdw'  window background color
      'gr_reg'  partially transparent region shading color
      'gr_acc'  accent for UI elements
      'gr_hlt'  highlight for selected elements
    Plot colors:
      'p0' to 'p7'  typically sampled from a ColorMap
    """
    def __init__(self, cmap=None, colors=None ):
        super().__init__()
        self.palette = fn.COLOR_REGISTRY.defaultColors()
        # self.palette  = { # populate dictionary of QColors with legacy defaults
        #     'b': QtGui.QColor(  0,  0,255,255), 'g': QtGui.QColor(  0,255,  0,255), 
        #     'r': QtGui.QColor(255,  0,  0,255), 'c': QtGui.QColor(  0,255,255,255),
        #     'm': QtGui.QColor(255,  0,255,255), 'y': QtGui.QColor(255,255,  0,255),
        #     'k': QtGui.QColor(  0,  0,  0,255), 'w': QtGui.QColor(255,255,255,255),
        #     'd': QtGui.QColor(150,150,150,255), 'l': QtGui.QColor(200,200,200,255), 
        #     's': QtGui.QColor(100,100,150,255)
        # }
        # self.addDefaultRamp()
        self.dark = None # is initially set when assigning system palette
        self.emulateSystem()
        self.cmap = None
        if cmap is not None: # prepare plot colors from provided colormap
            if isinstance(cmap, (str, colormap.ColorMap) ): # sampleColorMap will convert if needed
                self.sampleColorMap( cmap=cmap)
            if isinstance(cmap, (tuple,list)): # ('identifier', start, step)
                cmap, start, step = cmap
                self.sampleColorMap(cmap=cmap, start=start, step=step)
        if colors is not None:
            # print('color dictionary:', colors)
            self.add(colors) # override specified colors
            
    def __getitem__(self, key):
        """ Convenient shorthand access to palette colors """
        if isinstance(key, str): # access by color name
            return self.palette.get(key,None)
        if isinstance(key, int): # access by plot color index
            idx = key % 8 # map to 0 to 8
            key = 'p'+str(idx)
            return self.palette.get(key,None)
        return None

    def __setitem__(self, key, color):
        """ Convenient shorthand access to palette colors """
        if not isinstance(color, QtGui.QColor):
            color = QtGui.QColor(color)
            self.palette[key] = color 
        
    def colorMap(self):
        """
        Return the ColorMap object used to create plot colors or 'None' if not assigned.
        """
        return self.cmap

    def sampleColorMap(self, cmap=None, n_colors=8, prefix='p', start=0., step=None ):
        """
        Sample a ColorMap to update defined plot colors
        =============  ===============================================================================
        **Arguments**
        cmap           a ColorMap object to be sampled. If not given, a default color map is used
        n_colors       default '8': Number of assigned colors. 
                       The default set needs to include 'p0' to 'p7'
        prefix         default 'p' assigns colors as 'p0' to 'pXX', at least 'p7' is required.
                       Additional sets can be defined with a different prefix, e.g. 'col_0' to 'col_7'
                       All prefixes need to start with 'p' or 'col' to avoid namespace overlap with 
                       functional colors and hexadecimal numbers
        start          first sampled value (default is 0.000)
        step           step between samples. Default 'None' equally samples n colors from a 
                       linear colormap, including values 0.0 and 1.0.
                       Color values > 1. and < 0. wrap around!
        =============  ===============================================================================
        """
        if not (
            prefix[0]=='p' or
            prefix[0]=='m' or
            ( len(prefix)>=3 and prefix[:3]=='col')
        ):
            raise ValueError("'prefix' of plot color needs to start with 'p', 'm' or 'col'.")
        if cmap is None: 
            cmap = self.cmap
        if isinstance(cmap, str):
            cmap = colormap.get(cmap) # obtain ColorMap if identifier is given
        if cmap is None:
            raise ValueError("Please specify 'cmap' parameter when no default colormap is available.")
        if not isinstance( cmap, colormap.ColorMap ):
            raise ValueError("Failed to obtain ColorMap object for 'cmap' = '+str(cmap).")
            
        if prefix == 'p':
            self.cmap = cmap # replace default color map
            n_colors = 8 # always define 8 primary plot colors
        if step is None:
            step = 1 / (n_colors - 1) # sample 0. to 1. (inclusive) by default
        for cnt in range(n_colors):
            val = start + cnt * step
            # print( val )
            if val > 1.0 or val < 0.0: # don't touch 1.0 value
                val = val % 1. # but otherwise map to 0 to 1 range
            qcol = cmap[val]
            key = prefix + str(cnt)
            self.palette[key] = qcol
            
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
            
    def addDefaultRamp(self):
        """
        Adds a default ramp of grays m0 to m9,
        linearized according to CIElab lightness value
        """
        

    def emulateSystem(self):
        """
        Retrieves the current Qt 'active' palette and extracts the following colors:
        =====================================  ============================================================================
        'gr_fg','gr_txt' from QPalette.Text    (foreground color used with Base)
        'gr_bg'  from QPalette.Base            (background color for e.g. text entry widgets)
        'gr_wdw' from QPalette.Window          (a general background color)
        'gr_reg' from QPalette.AlternateBase   (alternating row background color)
        'gr_acc' from QPalette.Link            (color used for unvisited hyperlinks)
        'gr_hlt' from QPalette.Highlight       (color to indicate a selected item)
        =====================================  ============================================================================
        """
        app = QtWidgets.QApplication.instance()
        if app is None: return None        
        qPalette = app.palette()
        col_grp = QtGui.QPalette.Active
        colors = {}
        for key, alpha, col_role in (
            ('gr_bg' , None, QtGui.QPalette.Base),       # background color for e.g. text entry
            ('gr_fg' , None, QtGui.QPalette.WindowText), # overall foreground text color
            ('gr_txt', None, QtGui.QPalette.Text),       # foreground color used with Base
            ('gr_reg',  128, QtGui.QPalette.AlternateBase), # alternating row background color
            ('gr_acc', None, QtGui.QPalette.Link),       # color of unvisited hyperlink
            # ('gr_reg',  128, QtGui.QPalette.Highlight), # alternating row background color
            # ('gr_acc', None, QtGui.QPalette.Highlight),       # color of unvisited hyperlink
            ('gr_hlt', None, QtGui.QPalette.Highlight),  # color to indicate a selected item
            ('ui_wind', None, QtGui.QPalette.Window),    # a general background color
            ('ui_text', None, QtGui.QPalette.WindowText) #  overall foreground text color
        ):
            qcol = qPalette.color(col_grp, col_role)
            if alpha is not None: qcol.setAlpha(alpha)
            colors[key] = qcol
        self.add(colors)        
        colors = {
            'b': QtCore.Qt.blue  , 'c': QtCore.Qt.cyan, 'g': QtCore.Qt.green  , 
            'y': QtCore.Qt.yellow, 'r': QtCore.Qt.red , 'm': QtCore.Qt.magenta,
            'w': QtCore.Qt.white , 's': QtCore.Qt.gray, 'k': QtCore.Qt.black  ,
            'l': QtCore.Qt.lightGray, 'd': QtCore.Qt.darkGray
        }
        if not self.dark: # darken some colors for light mode
            colors['c'] = QtCore.Qt.darkCyan
            colors['g'] = QtCore.Qt.darkGreen
            colors['y'] = QtCore.Qt.darkYellow
            colors['m'] = QtCore.Qt.darkMagenta
        self.add(colors)
        colors = { 
            'p'+str(idx) : name 
            for idx, name in enumerate( ('gr_fg', 'y','r','m','b','c','g','s') ) 
        }
        self.add(colors)
        
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
        self.sampleColorMap( cmap=cmap, start=1.0, step=-1/8 ) # assign bright to dark, don't go all the way to background.
        # define colors 'm0' (near-black) to 'm8' (near-white):
        self.sampleColorMap( n_colors=10, cmap=cmap, step=1/9, prefix='m' ) 
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
        brightness = [
            (np.sum( ref_color * needed[key] ), key) 
            for key in needed
        ]
        brightness = sorted(brightness, key=lambda brightness: brightness[0])
        # print( brightness )
        avail = ('m3','m4','m5','m6','m7','m8')
        colors = {}
        for idx, (value, key) in enumerate(brightness):
            colors[key] = avail[idx]
        self.add( colors )
        
    def apply(self):
        """
        Applies this palette to the overall PyQtGraph color scheme.
        This provides the palette to NamedColorManager, which triggers a global refresh of named colors
        """
        fn.COLOR_REGISTRY.redefinePalette( colors=self.palette )
