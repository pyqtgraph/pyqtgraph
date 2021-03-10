from .Qt import QtGui, QtCore

from . import functions as fn # namedColorManager

__all__ = ['Palette']

LEGACY_RAW = { # legacy raw colors:
    'b': (  0,  0,255,255),
    'g': (  0,255,  0,255),
    'r': (255,  0,  0,255),
    'c': (  0,255,255,255),
    'm': (255,  0,255,255),
    'y': (255,255,  0,255),
    'k': (  0,  0,  0,255),
    'w': (255,255,255,255),
    'd': (150,150,150,255),
    'l': (200,200,200,255),
    's': (100,100,150,255)
}
LEGACY_FUNC = { # functional colors:
    'gr_fg'  : 'd',
    'gr_bg'  : 'k',
    'gr_txt' : 'd',
    'gr_acc' : (200,200,100,255),
    'gr_hov' : 'r',
    'gr_reg' : (  0,  0,255, 50)
}
LEGACY_PLOT = [ # plot / accent colors:
    'l','y','r','m','b','c','g','d','d','d','d','d'
]

MONOGREEN_RAW = {
    'col_g0':'#001000', 'col_g1':'#014801', 'col_g2':'#077110', 'col_g3':'#159326',
    'col_g4':'#2DB143', 'col_g5':'#50CD65', 'col_g6':'#7FE7A0', 'col_g7':'#BFFFD4'
}
MONOGREEN_FUNC = {
    'gr_fg'  : 'col_g5',
    'gr_bg'  : 'col_g0', # for distinction in testing, should be col_g0
    'gr_txt' : 'col_g5',
    'gr_acc' : 'col_g5',
    'gr_hov' : 'col_g7',
    'gr_reg' : ('col_g6', 30),
    # legacy colors:
    'b': 'col_g7', 'c': 'col_g6', 'g': 'col_g5', 
    'y': 'col_g4', 'r': 'col_g3', 'm': 'col_g2',
    'k': 'col_g1', 'w': 'col_g7',
    'd': 'col_g1', 'l': 'col_g4', 's': 'col_g7'
}
MONOGREEN_PLOT = [ # plot / accent colors:
    'col_g6', 'col_g4', 'col_g2', 'col_g7', 'col_g5', 'col_g3', 
    'col_g1', 'col_g3', 'col_g3', 'col_g3', 'col_g3', 'col_g3'
]

RELAXED_RAW = { # "fresh" raw colors:
    'col_orange':'#A64D21', 'col_l_orange':'#D98A62', 'col_d_orange':'#732E0B',
    'col_red'   :'#B32424', 'col_l_red'   :'#E66767', 'col_d_red'   :'#800D0D',
    'col_purple':'#991F66', 'col_l_purple':'#D956A3', 'col_d_purple':'#660A31',
    'col_violet':'#7922A6', 'col_l_violet':'#BC67E6', 'col_d_violet':'#5A0C80',
    'col_indigo':'#5F29CC', 'col_l_indigo':'#9673FF', 'col_d_indigo':'#380E8C',
    'col_blue'  :'#2447B3', 'col_l_blue'  :'#6787E6', 'col_d_blue'  :'#0D2980',
    'col_sky'   :'#216AA6', 'col_l_sky'   :'#77ADD9', 'col_d_sky'   :'#0B4473',
    'col_cyan'  :'#1C8C8C', 'col_l_cyan'  :'#73BFBF', 'col_d_cyan'  :'#095959',
    'col_green' :'#1F9952', 'col_l_green' :'#7ACC9C', 'col_d_green' :'#0A6630',
    'col_grass' :'#7AA621', 'col_l_grass' :'#BCD982', 'col_d_grass' :'#50730B',
    'col_yellow':'#BFB226', 'col_l_yellow':'#F2E985', 'col_d_yellow':'#80760D',
    'col_gold'  :'#A67A21', 'col_l_gold'  :'#D9B46C', 'col_d_gold'  :'#73500B',
    # 'col_black' :'#000000', 'col_gr1'     :'#242429', 'col_gr2'     :'#44444D',
    'col_black' :'#000000', 'col_gr1'     :'#161619', 'col_gr2'     :'#43434D',
    'col_gr3'   :'#70707F', 'col_gr4'     :'#9D9DB2', 'col_gr5'     :'#C9C9E5',
    'col_white' :'#FFFFFF'
}
RELAXED_DARK_FUNC= { # functional colors:
    'gr_fg'  : 'col_gr5', 
    'gr_bg'  : 'col_gr1',
    'gr_txt' : 'col_gr5', 
    'gr_acc' : 'col_cyan',
    'gr_hov' : 'col_white',
    'gr_reg' : ('col_cyan', 30),
    # legacy colors:
    'b': 'col_l_blue'  , 'c': 'col_l_cyan', 'g': 'col_l_green', 
    'y': 'col_l_yellow', 'r': 'col_l_red' , 'm': 'col_l_violet',
    'k': 'col_black'   , 'w': 'col_white',
    'd': 'col_gr2'     , 'l': 'col_gr4'   , 's': 'col_l_sky'
}
RELAXED_DARK_PLOT = [ # plot / accent colors:
    'col_l_sky'   , 
    'col_l_indigo', 
    'col_l_purple', 
    'col_l_red'   ,
    'col_l_gold'  , 
    'col_l_grass' ,
    'col_l_cyan'  , 
    'col_l_blue'  ,
    'col_l_violet',
    'col_l_orange', 
    'col_l_yellow', 
    'col_l_green' 
]

RELAXED_LIGHT_FUNC= { # functional colors:
    'gr_fg'  : 'col_gr1', 
    'gr_bg'  : 'col_gr5',
    'gr_txt' : 'col_black', 
    'gr_acc' : 'col_orange',
    'gr_hov' : 'col_black',
    'gr_reg' : ('col_blue', 30),
    # legacy colors:
    'b': 'col_blue'  , 'c': 'col_cyan', 'g': 'col_green', 
    'y': 'col_yellow', 'r': 'col_red' , 'm': 'col_violet',
    'k': 'col_black'   , 'w': 'col_white',
    'd': 'col_gr2'     , 'l': 'col_gr4'   , 's': 'col_sky'
}
RELAXED_LIGHT_PLOT = [ # plot / accent colors:
    'col_sky'   , 
    'col_indigo', 
    'col_purple', 
    'col_red'   ,
    'col_gold'  , 
    'col_grass' ,
    'col_cyan'  , 
    'col_blue'  ,
    'col_violet',
    'col_orange', 
    'col_yellow', 
    'col_green' 
]



def block_to_QColor( block, dic=None ):
    """ convert color information to a QColor """
    # allowed formats:
    # 'name'
    # ('name', alpha)
    # (R,G,B)   /   (R,G,B,alpha)
    alpha = None
    if isinstance(block, str): # return known QColor
        name = block
        if dic is None or name not in dic:
            if name[0] != '#':
                raise ValueError('Undefined color name '+str(block))
            return QtGui.QColor( name )
        else:
            return dic[name] 
    if not hasattr(block, '__len__'):
        raise ValueError('Invalid color definition '+str(block))
    if len(block) == 2:
        name, alpha = block
        if dic is None or name not in dic:
            if name[0] != '#':
                raise ValueError('Undefined color name '+str(block))
            qcol = QtGui.QColor( name )
        else:
            qcol = dic[ name ]
    elif len(block) in (3,4):
        qcol = QtGui.QColor( *block )
    
    if alpha is not None:
        qcol.setAlpha( alpha )
    return qcol            


def assemble_palette( raw_col, func_col, plot_col ):
    """
    assemble palette color dictionary from parts:
    raw_col should contain color information in (R,G,B,(A)) or hex format
    func_col typically contains keys of colors defined before
    plot_col is a list of plotting colors to be included as 'c0' to 'cX' (in hex)
    """
    pal = {}
    for part in [raw_col, func_col]:
        for key in part:
            col = part[key]
            pal[key] = block_to_QColor( col, pal )
    for idx, col in enumerate( plot_col ):
        key = 'p{:X}'.format(idx) # plot color 'pX' does not overlap hexadecimal codes.
        pal[key] = block_to_QColor( col, pal )
    return pal
    
DEFAULT_PALETTE = assemble_palette( LEGACY_RAW, LEGACY_FUNC, LEGACY_PLOT )

def get(name):
    if name == 'relaxed_dark':
        pal = assemble_palette( RELAXED_RAW, RELAXED_DARK_FUNC, RELAXED_DARK_PLOT )
    elif name == 'relaxed_light':
        pal = assemble_palette( RELAXED_RAW, RELAXED_LIGHT_FUNC, RELAXED_LIGHT_PLOT )
    elif name == 'monogreen':
        pal = assemble_palette( MONOGREEN_RAW, MONOGREEN_FUNC, MONOGREEN_PLOT )
    else:
        pal = DEFAULT_PALETTE
    return Palette( colors=pal )


class Palette(object):
    # minimum colors to be defined:
    def __init__(self, colors=None):
        super().__init__()
        self.palette = colors
    
    # needs: addColors
    # needs to be aware of number of plot colors
    # needs to be indexable by key and numerical plot color
    # indexed plot colors need to wrap around to work for any index.
    # needs: clearColors

    def apply(self):
        """
        provides palette to NamedColorManager, which triggers a global refresh of named colors
        """
        fn.NAMED_COLOR_MANAGER.redefinePalette( colors=self.palette )
