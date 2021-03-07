# from ..Qt import QtGui
from .Qt import QtGui, QtCore

from . import functions as fn

__all__ = ['NamedBrush']
DEBUG = False

class NamedBrush(QtGui.QBrush):
    """ Extends QPen to retain a functional color description """
    def __init__(self, name, alpha=None ):
        """
        Creates a new NamedBrush object.
        'name' should be in 'functions.Colors'
        'alpha' controls opacity which persists over palette changes
        """
        if DEBUG: print('  NamedBrush created as',name,alpha)
        super().__init__(QtCore.Qt.SolidPattern) # Initialize QBrush superclass
        self._identifier = (name, alpha)
        self._updateQColor(self._identifier)
        fn.NAMED_COLOR_MANAGER.register( self ) # manually register for callbacks

    def __eq__(self, other): # make this a hashable object
        # return other is self
        if isinstance(other, self.__class__):
            return self._identifier == other._identifier
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def __hash__(self):
        return id(self)

    def setColor(self, name=None, alpha=None):
        """ update color name. This does not trigger a global redraw. """
        if name is None:
            name = self._identifier[0]
        elif isinstance(name, QtGui.QColor):
            # Replicates alpha adjustment workaround in NamedPen, allowing only alpha to be adjusted retroactively
            if alpha is None:
                alpha = name.alpha() # extract from given QColor
            name = self._identifier[0]
            if DEBUG: print('  NamedBrush: setColor(QColor) call: set alpha to', alpha)
        self._identifier = (name, alpha)
        self._updateQColor(self._identifier)

    def setAlpha(self, alpha):
        """ update opacity value """
        self._identifier = (self._identifier[0], alpha)
        self._updateQColor(self._identifier)
        
    def _updateQColor(self, identifier, color_dict=None):
        """ update super-class QColor """
        name, alpha = identifier
        if color_dict is None:
            color_dict = fn.NAMED_COLOR_MANAGER.colors()
        try:
            qcol = fn.Colors[name]
        except ValueError as exc:
            raise ValueError("Color '{:s}' is not in list of defined colors".format(str(name)) ) from exc
        if alpha is not None:
            qcol.setAlpha( alpha )
        if DEBUG: print('  NamedBrush '+name+' updated to QColor ('+str(qcol.name())+', alpha='+str(alpha)+')')
        super().setColor( qcol )
        
    def identifier(self):
        """ return current color identifier """
        return self._identifier
        
    def paletteChange(self, color_dict):
        """ refresh QColor according to lookup of identifier in functions.Colors """
        if DEBUG: print('  NamedBrush: style change request:', self, type(color_dict))
        self._updateQColor(self._identifier, color_dict=color_dict)
        if DEBUG:
            qcol = super().color()
            name, alpha = self._identifier
            print('  NamedBrush: retrieved new QColor ('+str(qcol.name())+') '
                + 'for name '+str(name)+' ('+str(alpha)+')' )
