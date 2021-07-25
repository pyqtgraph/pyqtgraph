from .Qt import QtCore, QtGui


# import numpy as np
import weakref
import itertools
# import collections
# import warnings
# import time
from .palette import Palette
from . import functions as fn

__all__ = ['StyleRegistry']

# An instantiated QObject is required to emit QSignals. 
# functions.py initializes and maintains STYLE_REGISTRY for this purpose.
class StyleRegistry(QtCore.QObject):
    """
    Provides style change signals and provides access to the current palette
    Instantiated by 'functions.py' and retrievable as functions.STYLE_REGISTRY
    """
    graphStyleChanged = QtCore.Signal() # equated to pyqtSignal in qt.py for PyQt
    _registrationGenerator = itertools.count()

    def __init__(self):
        """ initialization """
        super().__init__()
       
        # needs to be externally assigned :
        # importing palette (-> colormap -> functions) creates a circular import.
        self._palette = Palette('legacy')
        
        # set of objects that are registered for update on palette change:
        self.registered_objects = {} # stores registration: (weakref(object), descriptor), deletion handled by weakref.finalize
        # DEBUG
        # self.registered_history = {} # stores all previous registrations for debugging
        self.color_cache = weakref.WeakValueDictionary() # keeps a cache of QColor objects, key: (name, alpha)
        self.pen_cache   = weakref.WeakValueDictionary() # keeps a cache of QPen   objects, key: (name, width, alpha)
        self.brush_cache = weakref.WeakValueDictionary() # keeps a cache of QBrush objects, key: (name, alpha)

    def palette(self):
        """
        Returns the currently active palette
        """
        return self._palette
        
        
    def getPaletteColor(self, identifier, alpha=None):
        """
        Returns a color from the currently active palette
        """    
        if identifier not in self._palette._colors:
            raise ValueError(f"Identifier '{identifier}' is not part of current palette.")
        qcol = QtGui.QColor( self._palette._colors[identifier] )
        if alpha is not None: qcol.setAlphaF(alpha)
        return qcol
        
    def samplePlotColor(self, idx, num=9, alpha=None):
        """
        Returns a sampled color from the plot color map

        Parameters
        ----------
        idx   : (int) plot color index
        num   : (int, optional) total number of sample steps, default to 9.
        alpha : (float, optional) overriding alpha value
        """
        qcol = self._palette.map_p[idx/num]
        if alpha is not None: qcol.setAlphaF(alpha)
        return qcol

    def sampleMonoColor(self, value, alpha=None):
        """
        Returns a sampled color from the monochrome color map

        Parameters
        ----------
        value : (float) plot color index 0.0-1.0
        alpha : (float, optional) overriding alpha value
        """
        qcol = self._palette.map_m[value]
        if alpha is not None: qcol.setAlphaF(alpha)
        return qcol

    # def getRegisteredColor( self, color_desc, skipCache=False ):
    #     """ 
    #     Returns a registered QColor according to (name, alpha) color descriptor
    #     Setting skipCache = True creates a new registered QColor that is not added to the cache
    #     Otherwise the color is cached and reused on the next request with the same pen id.
    #     """
    #     register = True # register, unless this is a hex code color
    #     # print('color descriptor:',color_desc)
    #     desc = self.confirmColorDescriptor(color_desc)
    #     if desc is False: return None # not a valid color id
    #     name, alpha = desc
    #     if name is None:
    #         return QtGui.QColor(0, 0, 0, 0) # fully transparent
    #     if name[0] == '#': # skip cache and registry for fixed hex colors
    #         name, alpha_from_hex = _expand_rgba_hex_string( name )
    #         if name is None: return None
    #         if alpha is None and alpha_from_hex is not None:
    #             # alpha = alpha_from_hex
    #             desc = (name, alpha_from_hex) # handle extended 4 bytes rgba hex strings
    #         # print('preparing hex color:',name)
    #         skipCache = True
    #         register  = False
    #     elif name not in self.color_dic:
    #         warnings.warn(f"Unknown color identifier '{name}' encountered.")
    #         return None # unknown color identifier
    #     if not skipCache and desc in self.color_cache:
    #         return self.color_cache[desc]
    #     qcol = QtGui.QColor()
    #     self._update_QColor(qcol, desc)
    #     if register: self.register(qcol, desc) # register for updates on palette change
    #     if not skipCache: # skipCache disable both reading and writing cache
    #         self.color_cache[desc] = qcol
    #     return qcol
        
    # def getRegisteredPen( self, pen_desc, skipCache=False ):
    #     """
    #     Returns a registered QPen according to (name, width, alpha) pen descriptor
    #     Setting skipCache = True creates a new registered QPen that is not added to the cache.
    #     Otherwise the pen is cached and reused on the next request with the same pen descriptor.
    #     """
    #     register = True # register, unless this is a hex code color
    #     # print('pen descriptor:',pen_desc)
    #     desc = self.confirmPenDescriptor(pen_desc)
    #     if desc is False: return None # not a valid pen id
    #     name, width, alpha = desc
    #     if name is None:
    #         return QtGui.QPen( QtCore.Qt.PenStyle.NoPen )
    #     if name[0] == '#': # skip cache and registry for fixed hex colors
    #         name, alpha_from_hex = _expand_rgba_hex_string( name )
    #         if name is None: return None
    #         if alpha is None and alpha_from_hex is not None:
    #             # alpha = alpha_from_hex
    #             desc = (name, width, alpha_from_hex) # handle extended 4 bytes rgba hex strings
    #         # print('preparing hex pen:',name)
    #         skipCache = True
    #         register  = False
    #     elif name not in self.color_dic:
    #         warnings.warn(f"Unknown color identifier '{name}' encountered in pen descriptor.")
    #         return None # unknown color identifier
    #     if not skipCache and desc in self.pen_cache:
    #         return self.pen_cache[desc]
    #     qpen = QtGui.QPen()
    #     self._update_QPen(qpen,desc)
    #     if register: self.register( qpen, desc ) # register for updates on palette change
    #     if not skipCache: # skipCache disable both reading and writing cache
    #         self.pen_cache[desc] = qpen
    #     return qpen

    # def getRegisteredBrush( self, brush_desc, skipCache=False ):
    #     """
    #     Returns a registered QBrush according to (name, alpha) brush descriptor
    #     Setting skipCache = True creates a new registered QBrush that is not added to the cache.
    #     Otherwise the brush is cached and reused on the next request with the same descriptor.
    #     """
    #     register = True # register, unless this is a hex code color
    #     # print('brush descriptor:',brush_desc)
    #     desc = self.confirmColorDescriptor(brush_desc) # brush id is the same (name, alpha) format as color id
    #     if desc is False: return None # not a valid brush id
    #     name, alpha = desc
    #     if name is None:
    #         return QtGui.QBrush( QtCore.Qt.BrushStyle.NoBrush ) 
    #     if name[0] == '#': # skip cache and registry for fixed hex colors
    #         name, alpha_from_hex = _expand_rgba_hex_string( name )
    #         if name is None: return None
    #         if alpha is None and alpha_from_hex is not None:
    #             # alpha = alpha_from_hex
    #             desc = (name, alpha_from_hex) # handle extended 4 bytes rgba hex strings
    #         skipCache = True
    #         register  = False
    #     elif name not in self.color_dic:
    #         warnings.warn(f"Unknown color identifier '{name}' encountered in brush descriptor.")
    #         return None # unknown color identifier
    #     if not skipCache and desc in self.brush_cache:
    #         return self.brush_cache[desc]
    #     qbrush = QtGui.QBrush(QtCore.Qt.BrushStyle.SolidPattern) # make sure this brush fills once color is set
    #     self._update_QBrush(qbrush,desc)
    #     if register: self.register( qbrush, desc ) # register for updates on palette change
    #     if not skipCache: # skipCache disable both reading and writing cache
    #         self.brush_cache[desc] = qbrush
    #     return qbrush
        
    # def _update_QColor(self, qcol, desc):
    #     """ updates qcol to match descriptor for current palette """
    #     # print('updating color to match',desc)
    #     name, alpha = desc
    #     if name[0] != '#':
    #         qcol.setRgba( self.color_dic[name].rgba() )
    #     else:
    #         qcol.setNamedColor(name) # set from hex string
    #     if alpha is not None: qcol.setAlpha(alpha)
    #     # print('= hex:', qcol.name() )

    # def _update_QPen(self, qpen, desc):
    #     """ updates qpen to match descriptor for current palette """
    #     # print('updating pen to match',desc)
    #     name, width, alpha = desc
    #     if name[0] != '#':
    #         if alpha is None:
    #             qpen.setColor( self.color_dic[name] ) # automatically copies
    #         else:
    #             qcol = QtGui.QColor(self.color_dic[name]) # make a copy
    #             qcol.setAlpha(alpha)
    #             qpen.setColor(qcol)
    #     else: # set from hex string instead:
    #         qcol = QtGui.QColor(name)
    #         if alpha is not None: qcol.setAlpha(alpha)
    #         qpen.setColor(qcol)
    #     if width is not None:
    #         qpen.setWidth(width)
    #     # print('= hex:', qpen.color().name() )

    # def _update_QBrush(self, qbrush, desc):
    #     """ updates qbrush to match descriptor for current palette """
    #     # print('updating brush',qbrush,'to match',desc)
    #     name, alpha = desc
    #     if name[0] != '#':
    #         if alpha is None:
    #             qbrush.setColor( self.color_dic[name] ) # automatically copies
    #         else:
    #             qcol = QtGui.QColor(self.color_dic[name]) # make a copy
    #             qcol.setAlpha(alpha)
    #             qbrush.setColor(qcol)
    #     else: # set from hex string instead:
    #         qcol = QtGui.QColor(name)
    #         if alpha is not None: qcol.setAlpha(alpha)
    #         qbrush.setColor(qcol)
    #     # print('= hex:', qbrush.color().name(), qbrush.color().alpha() )

    # def register(self, obj, desc):
    #     """
    #     Registers obj (QColor, QPen or QBrush) to be updated according to the descriptor on palette change
    #     """
    #     if hasattr(obj,'colorRegistration'):
    #         registration = obj.registration
    #         if registration in self.registered_objects:
    #             # if this object is already registered, we should not try to add another finalize call.
    #             return
    #     else:
    #         registration = next(StyleRegistry._registrationGenerator)
    #         obj.registration = registration # patch in attribute
    #     self.registered_objects[registration] = (weakref.ref(obj), desc)
    #     # DEBUG:
    #     # self.registered_history[registration] = (weakref.ref(obj), desc)
    #     fin = weakref.finalize(obj, self.unregister, registration )
    #     fin.atexit = False # no need to clean up registry on program exit
    #     # DEBUG:
    #     # print('registering', registration, '(',str(obj),'):',str(desc))

    # def unregister(self, registration):
    #     """
    #     Removes obj (QColor, QPen or QBrush) from the registry, usually called by finalize on deletion
    #     """
    #     obj, desc = self.registered_objects.pop(registration, (False, False))        
    #     if obj is False:
    #         # DEBUG:
    #         # obj_hist, desc_hist = self.registered_history.get(registration, (False, False))
    #         # if obj_hist is not False:
    #         #     raise RuntimeError(f'Unregistered object {registration} already unregistered, previously registered as {obj_hist} ({desc_hist}).')
    #         raise RuntimeError(f'Unregistered object {registration} unlisted.')

    #     # DEBUG:
    #     # print('unregistering', registration, '(',str(obj),'):',str(desc))
    #     del obj, desc

    # def redefinePalette(self, colors=None):
    #     """ 
    #     Update list of registered colors if 'colors' dictionary is given
    #     Emits paletteHasChanged signals to color objects and widgets, even if color_dict is None 
    #     """
    #     if colors is not None:
    #         for key in DEFAULT_COLORS:
    #             if key not in colors:
    #                 raise ValueError(f"Palette definition is missing '{key}'")
    #         self.color_dic.clear()
    #         self.color_dic.update(colors)

    #     # notifies registered color objects of new assignments:
    #     for ref, desc in self.registered_objects.values():
    #         # ref, desc = self.registered_objects[key]
    #         obj = ref()
    #         # print('updating', obj)
    #         if obj is None:
    #             warnings.warn(f"Expired object with descriptor '{desc})' remains in color registry.", RuntimeWarning)
    #         elif isinstance(obj, QtGui.QColor):
    #             self._update_QColor(obj, desc)
    #         elif isinstance(obj, QtGui.QPen):
    #             self._update_QPen(obj, desc)
    #         elif isinstance(obj, QtGui.QBrush):
    #             self._update_QBrush(obj, desc)
    #     # notify all graphics widgets that redraw is required:
    #     self.graphStyleChanged.emit()


print('initializing style registry')
STYLE_REGISTRY = StyleRegistry()
print('registering with functions')
fn.STYLE_REGISTRY = STYLE_REGISTRY
