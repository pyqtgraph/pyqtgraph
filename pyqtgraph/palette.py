from . import Qt
from .Qt import QtCore, QtGui, QtWidgets

from . import functions as fn # namedColorManager
from . import colormap

__all__ = ['Palette']

#### todo list ####
# define legacy colors for relaxed-dark
# find color definitions for relaxed-light
# define color names for relaxed palettes
# enable color adjustment in PaletteTestandEdit.py!

PALETTE_DEFINITIONS = {
    'legacy': {
        'colormap_sampling' : None,
        'b': (  0,  0,255,255), 'g': (  0,255,  0,255), 'r': (255,  0,  0,255), 
        'c': (  0,255,255,255), 'm': (255,  0,255,255), 'y': (255,255,  0,255),
        'k': (  0,  0,  0,255), 'w': (255,255,255,255),
        'd': (150,150,150,255), 'l': (200,200,200,255), 's': (100,100,150,255),
        # --- functional colors ---
        'gr_fg' : 'd', 'gr_bg' : 'k',
        'gr_txt': 'd', 'gr_acc': (200,200,100,255),
        'gr_hlt': 'r', 'gr_reg': (  0,  0,255,100),
        # --- manually assigned plot colors ---
        'p0':'l', 'p1':'y', 'p2':'r', 'p3':'m',
        'p4':'b', 'p5':'c', 'p6':'g', 'p7':'d'
    },
    'relaxed_dark':{
        'colormap_sampling': ('CET-C6', 0.430, -0.125),
        'col_black' :'#000000', 'col_white' :'#FFFFFF',
        'col_gr1':'#19232D', 'col_gr2':'#32414B', 'col_gr3':'#505F69', # match QDarkStyle background colors 
        'col_gr4':'#787878', 'col_gr5':'#AAAAAA', 'col_gr6':'#F0F0F0', # match QDarkstyle foreground colors
        # --- functional colors ---
        'gr_fg' : 'col_gr4', 'gr_bg' : 'col_gr1',
        'gr_txt': 'col_gr5', 'gr_acc': '#1464A0',  #col_cyan',
        'gr_hlt': 'col_white', 'gr_reg': ('#1464A0',100)
        # legacy colors:
        # 'b': 'col_l_blue'  , 'c': 'col_l_cyan', 'g': 'col_l_green', 
        # 'y': 'col_l_yellow', 'r': 'col_l_red' , 'm': 'col_l_violet',
        # 'k': 'col_black'   , 'w': 'col_white',
        # 'd': 'col_gr2'     , 'l': 'col_gr4'   , 's': 'col_l_sky'
    },
    'synthwave':{
        'colormap_sampling': ('CET-L8', 0.275, 0.100),
        'col_black' :'#000000', 'col_white' :'#FFFFFF',
        'col_gr1':'#19232D', 'col_gr2':'#32414B', 'col_gr3':'#505F69', # match QDarkStyle background colors 
        'col_gr4':'#787878', 'col_gr5':'#AAAAAA', 'col_gr6':'#F0F0F0', # match QDarkstyle foreground colors
        # --- functional colors ---
        'gr_fg' : 'col_gr4', 'gr_bg' : 'col_gr1',
        'gr_txt': 'col_gr5', 'gr_acc': '#1464A0',  #col_cyan',
        'gr_hlt': 'col_white', 'gr_reg': ('#1464A0',100)
        # legacy colors:
        # 'b': 'col_l_blue'  , 'c': 'col_l_cyan', 'g': 'col_l_green', 
        # 'y': 'col_l_yellow', 'r': 'col_l_red' , 'm': 'col_l_violet',
        # 'k': 'col_black'   , 'w': 'col_white',
        # 'd': 'col_gr2'     , 'l': 'col_gr4'   , 's': 'col_l_sky'
    }

}

# RELAXED_RAW = { # "fresh" raw colors:
#     'col_orange':'#A64D21', 'col_l_orange':'#D98A62', 'col_d_orange':'#732E0B',
#     'col_red'   :'#B32424', 'col_l_red'   :'#E66767', 'col_d_red'   :'#800D0D',
#     'col_purple':'#991F66', 'col_l_purple':'#D956A3', 'col_d_purple':'#660A31',
#     'col_violet':'#7922A6', 'col_l_violet':'#BC67E6', 'col_d_violet':'#5A0C80',
#     'col_indigo':'#5F29CC', 'col_l_indigo':'#9673FF', 'col_d_indigo':'#380E8C',
#     'col_blue'  :'#2447B3', 'col_l_blue'  :'#6787E6', 'col_d_blue'  :'#0D2980',
#     'col_sky'   :'#216AA6', 'col_l_sky'   :'#77ADD9', 'col_d_sky'   :'#0B4473',
#     'col_cyan'  :'#1C8C8C', 'col_l_cyan'  :'#73BFBF', 'col_d_cyan'  :'#095959',
#     'col_green' :'#1F9952', 'col_l_green' :'#7ACC9C', 'col_d_green' :'#0A6630',
#     'col_grass' :'#7AA621', 'col_l_grass' :'#BCD982', 'col_d_grass' :'#50730B',
#     'col_yellow':'#BFB226', 'col_l_yellow':'#F2E985', 'col_d_yellow':'#80760D',
#     'col_gold'  :'#A67A21', 'col_l_gold'  :'#D9B46C', 'col_d_gold'  :'#73500B',
#     # 'col_black' :'#000000', 'col_gr1'     :'#242429', 'col_gr2'     :'#44444D',
#     'col_black' :'#000000', 'col_gr1'     :'#161619', 'col_gr2'     :'#43434D',
#     'col_gr3'   :'#70707F', 'col_gr4'     :'#9D9DB2', 'col_gr5'     :'#C9C9E5',
#     'col_white' :'#FFFFFF'
# }
# RELAXED_DARK_FUNC= { # functional colors:
#     'gr_fg'  : 'col_gr5', 
#     'gr_bg'  : 'col_gr1',
#     'gr_txt' : 'col_gr5', 
#     'gr_acc' : 'col_cyan',
#     'gr_hlt' : 'col_white',
#     'gr_reg' : ('col_cyan',100),
#     # legacy colors:
#     'b': 'col_l_blue'  , 'c': 'col_l_cyan', 'g': 'col_l_green', 
#     'y': 'col_l_yellow', 'r': 'col_l_red' , 'm': 'col_l_violet',
#     'k': 'col_black'   , 'w': 'col_white',
#     'd': 'col_gr2'     , 'l': 'col_gr4'   , 's': 'col_l_sky'
# }
# RELAXED_DARK_PLOT = [ # plot / accent colors:
#     'col_l_sky'   , 
#     'col_l_indigo', 
#     'col_l_purple', 
#     'col_l_red'   ,
#     'col_l_gold'  , 
#     'col_l_grass' ,
#     'col_l_cyan'  , 
#     'col_l_blue'  ,
#     'col_l_violet',
#     'col_l_orange', 
#     'col_l_yellow', 
#     'col_l_green' 
# ]

# RELAXED_LIGHT_FUNC= { # functional colors:
#     'gr_fg'  : 'col_gr1', 
#     'gr_bg'  : 'col_gr5',
#     'gr_txt' : 'col_black', 
#     'gr_acc' : 'col_orange',
#     'gr_reg' : ('col_blue',100),
#     # legacy colors:
#     'b': 'col_blue'  , 'c': 'col_cyan', 'g': 'col_green', 
#     'y': 'col_yellow', 'r': 'col_red' , 'm': 'col_violet',
#     'k': 'col_black'   , 'w': 'col_white',
#     'd': 'col_gr2'     , 'l': 'col_gr4'   , 's': 'col_sky'
# }
# RELAXED_LIGHT_PLOT = [ # plot / accent colors:
#     'col_sky'   , 
#     'col_indigo', 
#     'col_purple', 
#     'col_red'   ,
#     'col_gold'  , 
#     'col_grass' ,
#     'col_cyan'  , 
#     'col_blue'  ,
#     'col_violet',
#     'col_orange', 
#     'col_yellow', 
#     'col_green' 
# ]



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
        self.palette  = { # populate dictionary of QColors with legacy defaults
            'b': QtGui.QColor(  0,  0,255,255), 'g': QtGui.QColor(  0,255,  0,255), 
            'r': QtGui.QColor(255,  0,  0,255), 'c': QtGui.QColor(  0,255,255,255),
            'm': QtGui.QColor(255,  0,255,255), 'y': QtGui.QColor(255,255,  0,255),
            'k': QtGui.QColor(  0,  0,  0,255), 'w': QtGui.QColor(255,255,255,255),
            'd': QtGui.QColor(150,150,150,255), 'l': QtGui.QColor(200,200,200,255), 
            's': QtGui.QColor(100,100,150,255)
        }
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
        valid = prefix[0]=='p' or ( len(prefix)>=3 and prefix[:3]=='col')
        if not valid:
            raise ValueError("'prefix' of plot color needs to start with 'p'.")
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
            ('gr_reg',  100, QtGui.QPalette.AlternateBase), # alternating row background color
            ('gr_acc', None, QtGui.QPalette.Link),       # color of unvisited hyperlink
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
        cmap = colormap.makeMonochrome(color)
        if cmap is None: 
            raise ValueError("Failed to generate color for '"+str(color)+"'")
        self.sampleColorMap( cmap=cmap, start=1.0, step=-1/8 ) # assign bright to dark, don't go all the way to background.
        # define colors 'm0' (near-black) to 'm8' (near-white):
        self.sampleColorMap( n_colors=9, cmap=cmap, step=1/8, prefix='col_m' ) 
        colors =  {
            'gr_bg' : 'col_m0', 'gr_fg' : 'col_m4',
            'gr_txt': 'col_m5', 'gr_acc': 'col_m6',
            'gr_hlt': 'col_m7', 'gr_reg': ('col_m1', 30),
            'k': 'col_m0', 'd': 'col_m1', 's': 'col_m3', 'l': 'col_m6', 'w': 'col_m7'
        }
        self.add( colors )
        # make a dictionary of plot colors (except darkest and lightest) to emulate primary colors:
        avail  = { key: self.palette[key] for key in ('p0','p1','p2','p3','p4','p5','p6') }
        needed = { # int RGB colors that we are looking to emulate:
            'b': (  0,  0,255), 'c': (  0,255,255), 'g': (  0,255,  0),
            'y': (255,255,  0), 'r': (255,  0,  0), 'm': (255,  0,255)
        }
        colors = {}
        for nd_key in needed:
            nd_tup = needed[nd_key] # int RGB tuple to be represented
            best_dist = 1e10
            best_key = None
            for av_key in avail:
                av_tup = avail[av_key].getRgb() # returns (R,G,B,A) tuple
                sq_dist = (nd_tup[0]-av_tup[0])**2 + (nd_tup[1]-av_tup[1])**2 + (nd_tup[2]-av_tup[2])**2
                if sq_dist < best_dist:
                    best_dist = sq_dist
                    best_key  = av_key
            # print('assigning',nd_key,'as',best_key,':',avail[best_key].getRgb() )
            colors[nd_key] = avail[best_key]
            del avail[best_key] # remove from available list
        self.add( colors )

    def apply(self):
        """
        Applies this palette to the overall PyQtGraph color scheme.
        This provides the palette to NamedColorManager, which triggers a global refresh of named colors
        """
        fn.NAMED_COLOR_MANAGER.redefinePalette( colors=self.palette )
