# from ..Qt import QtGui
from .Qt import QtGui, QtCore

from . import functions as fn

__all__ = ['NamedPen']
DEBUG = False

class NamedPen(QtGui.QPen):
    """ Extends QPen to retain a functional color description """
    def __init__(self, name, width=1, alpha=None ):
        """
        Creates a new NamedPen object.
        'identifier' should be a 'name' included in 'functions.Colors' or
        '(name, alpha)' to include transparency
        """
        try:
            qcol = fn.Colors[name]
            # print('QColor alpha is', qcol.alpha() )
        except ValueError as exc:
            raise ValueError("Color {:s} is not in list of defined colors".format(str(name)) ) from exc
        if alpha is not None: 
            if DEBUG: print('  NamedPen: setting alpha to',alpha)
            qcol.setAlpha( alpha )

        super().__init__( QtGui.QBrush(qcol), width) # Initialize QPen superclass
        super().setCosmetic(True)
        self._identifier = (name, alpha)
        fn.NAMED_COLOR_MANAGER.register( self ) # manually register for callbacks

    def __eq__(self, other): # make this a hashable object
        return other is self
    def __hash__(self):
        return id(self)

    # def _parse_identifier(self, identifier):
    #     """ parse identifier parameter, which can be 'name' or '(name, alpha)' """
    #     alpha = None
    #     if isinstance(identifier, str):
    #         name = identifier
    #     else:
    #         try:
    #             name, alpha = identifier
    #         except ValueError as exc:
    #             raise ValueError("Invalid argument. 'identifier' should be 'name' or ('name', 'alpha'), but is {:s}".format(str(color)) ) from exc
    #     if name[0] == '#':
    #         raise TypeError("NamedPen should not be used for fixed colors ('name' = {:s} was given)".format(str(name)) )
    #     return name, alpha

    def setColor(self, name=None, alpha=None):
        """ update color name. This does not trigger a global redraw. """
        if name is None:
            name = self._identifier[0]
        elif isinstance(name, QtGui.QColor):
            # this is a workaround for the alpha adjustements in AxisItem:
            # While the color will not change, the alpha value can be adjusted as needed.
            alpha = name.alpha() # extract
            self._identifier = self._identifier[0], alpha
            # print('  NamedColor setColor(QColor) call: set alpha to', name.alpha())
            name = self._identifier[0]
        try:
            qcol = fn.Colors[name]
        except ValueError as exc:
            raise ValueError("Color {:s} is not in list of defined colors".format(str(name)) ) from exc
        if alpha is not None: 
            qcol.setAlpha( alpha )
        super().setColor( qcol )
        self._identifier = (name, alpha)
        

    # def setBrush(self):
    #     """ disabled """
    #     return None

    def paletteChange(self, color_dict):
        """ refresh QColor according to lookup of identifier in functions.Colors """
        if DEBUG: print('  NamedPen: style change request:', self, type(color_dict))
        name, alpha = self._identifier
        if color_dict is None: # manually retrieve color manager palette
            color_dict = fn.NAMED_COLOR_MANAGER.colors()
        try:
            qcol = color_dict[name]
            if DEBUG: print('  NamedPen: retrieved new QColor (', qcol.getRgb(), ') for name', name)
        except ValueError as exc:
            raise ValueError("Color {:s} is not in list of defined colors".format(str(name)) ) from exc
        if alpha is not None:
            qcol.setAlpha( alpha )
        super().setColor(qcol)
