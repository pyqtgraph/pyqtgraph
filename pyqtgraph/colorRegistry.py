from .Qt import QtCore, QtGui

import weakref
import itertools
import warnings

__all__ = ['ColorRegistry']
DEBUG = False

DEFAULT_COLORS = {
    'b': QtGui.QColor(  0,  0,255,255),
    'g': QtGui.QColor(  0,255,  0,255),
    'r': QtGui.QColor(255,  0,  0,255),
    'c': QtGui.QColor(  0,255,255,255),
    'm': QtGui.QColor(255,  0,255,255),
    'y': QtGui.QColor(255,255,  0,255),
    'k': QtGui.QColor(  0,  0,  0,255),
    'w': QtGui.QColor(255,255,255,255),
    'd': QtGui.QColor(150,150,150,255),
    'l': QtGui.QColor(200,200,200,255),
    's': QtGui.QColor(100,100,150,255),
    'gr_acc':QtGui.QColor(200,200,100,255), # graphical accent color: pastel yellow
    'gr_reg':QtGui.QColor(  0,  0,255, 50)  # graphical region marker: translucent blue
}
for key, col in [ # add functional colors
    ('gr_fg','d'),  # graphical foreground
    ('gr_bg','k'),  # graphical background
    ('gr_txt','d'), # graphical text color
    ('gr_hlt','r')  # graphical hover color
]:
    DEFAULT_COLORS[key] = DEFAULT_COLORS[col]

for idx, col in enumerate( ( # twelve predefined plot colors
    'l','y','r','m','b','c','g','d'
) ): 
    key = 'p{:X}'.format(idx)
    DEFAULT_COLORS[key] = DEFAULT_COLORS[col]
del key
    
# An instantiated QObject is required to emit QSignals. 
# functions.py initializes and maintains COLOR_REGISTRY for this purpose.
class ColorRegistry(QtCore.QObject):
    """
    Provides palette change signals and maintains color name dictionary
    Typically instantiated by functions.py as COLOR_REGISTRY
    Instantiated by 'functions.py' and retrievable as functions.COLOR_REGISTRY
    """
    paletteHasChangedSignal = QtCore.Signal() # equated to pyqtSignal in qt.py for PyQt
    
    _registrationGenerator = itertools.count()

    def __init__(self, color_dic):
        """ initialization """
        super().__init__()
        self.color_dic = color_dic # this is the imported functions.Colors!
        self.color_dic.clear()
        self.color_dic.update( DEFAULT_COLORS)
        # set of objects that are registered for update on palette change:
        self.registered_objects = {} # stores registration: (weakref(object), descriptor), deletion handled by weakref.finalize
        self.color_cache = weakref.WeakValueDictionary() # keeps a cache of QColor objects, key: (name, alpha)
        self.pen_cache   = weakref.WeakValueDictionary() # keeps a cache of QPen   objects, key: (name, width, alpha)
        self.brush_cache = weakref.WeakValueDictionary() # keeps a cache of QBrush objects, key: (name, alpha)

    def confirmColorDescriptor(self, color):
        """
        Takes a color or brush argument and tries to convert it to a normalized descriptor in the form
        ``( str(name), None/int(alpha) )``. Returns False if this is not possible.
        """
        name = alpha = None
        if color is None or isinstance( color, str):
            name = color
        elif hasattr( color, '__len__'):
            length = len(color)
            if length >= 1: name  = color[0]
            if length >= 2: alpha = color[1]
        if name is not None:
            if not isinstance(name,str): return False # pen id has to be str or a list-like starting with str
            if len(name) < 1: name = None # map '' to None
            # elif name[0] == '#': return False # strings that start with # are explicit rgb codes
        if alpha is not None: alpha = int(alpha)
        return (name, alpha)

    def confirmPenDescriptor(self, pen):
        """ 
        Takes a pen argument and tries to convert it to a normalized descriptor in the form
        ``( str(name), None/int(width), None/int(alpha) )``. Returns False if this is not possible.
        """
        name = width = alpha = None
        if pen is None or isinstance( pen, str):
            name = pen
        elif hasattr( pen, '__len__'):
            length = len(pen)
            if length >= 1: name  = pen[0]
            if length >= 2: width = pen[1]
            if length >= 3: alpha = pen[2]
        if name  is not None: 
            if type(name) != str: return False # pen id has to be str or a list-like starting with str
            if len(name) < 1 : name = None # map '' to None
            # elif name[0] == '#'   : return False # strings that start with # are explicit rgb codes
        if width is not None: width = int(width)
        if alpha is not None: alpha = int(alpha)
        return (name, width, alpha)

    def getRegisteredColor( self, color_desc, skipCache=False ):
        """ 
        Returns a registered QColor according to (name, alpha) color descriptor
        Setting skipCache = True creates a new registered QColor that is not added to the cache
        Otherwise the color is cached and reused on the next request with the same pen id.
        """
        register = True # register, unless this is a hex code color
        # print('color descriptor:',color_desc)
        desc = self.confirmColorDescriptor(color_desc)
        if desc is False: return None # not a valid color id
        name, alpha = desc
        if name is None:
            return QtGui.QColor(0, 0, 0, 0) # fully transparent
        if name[0] == '#': # skip cache and registry for fixed hex colors
            # print('preparting hex color:',name)
            skipCache = True
            register  = False
        if not skipCache and desc in self.color_cache:
            return self.color_cache[desc]
        qcol = QtGui.QColor()
        self._update_QColor(qcol, desc)
        if register: self.register(qcol, desc) # register for updates on palette change
        if not skipCache: # skipCache disable both reading and writing cache
            self.color_cache[desc] = qcol
        return qcol
        
    def getRegisteredPen( self, pen_desc, skipCache=False ):
        """
        Returns a registered QPen according to (name, width, alpha) pen descriptor
        Setting skipCache = True creates a new registered QPen that is not added to the cache.
        Otherwise the pen is cached and reused on the next request with the same pen descriptor.
        """
        register = True # register, unless this is a hex code color
        # print('pen descriptor:',pen_desc)
        desc = self.confirmPenDescriptor(pen_desc)
        if desc is False: return None # not a valid pen id
        name, width, alpha = desc
        if name is None:
            return QtGui.QPen( QtCore.Qt.NoPen )
        if name[0] == '#': # skip cache and registry for fixed hex colors
            # print('preparting hex pen:',name)
            skipCache = True
            register  = False
        if not skipCache and desc in self.pen_cache:
            return self.pen_cache[desc]
        qpen = QtGui.QPen()
        self._update_QPen(qpen,desc)
        if register: self.register( qpen, desc ) # register for updates on palette change
        if not skipCache: # skipCache disable both reading and writing cache
            self.pen_cache[desc] = qpen
        return qpen

    def getRegisteredBrush( self, brush_desc, skipCache=False ):
        """
        Returns a registered QBrush according to (name, alpha) brush descriptor
        Setting skipCache = True creates a new registered QBrush that is not added to the cache.
        Otherwise the brush is cached and reused on the next request with the same descriptor.
        """
        register = True # register, unless this is a hex code color
        # print('brush descriptor:',brush_desc)
        desc = self.confirmColorDescriptor(brush_desc) # brush id is the same (name, alpha) format as color id
        if desc is False: return None # not a valid brush id
        name, alpha = desc
        if name is None:
            # print('returning blank brush!')
            return QtGui.QBrush( QtCore.Qt.NoBrush ) 
        if name[0] == '#': # skip cache and registry for fixed hex colors
            # print('preparing hex brush:',name)
            skipCache = True
            register  = False
        if not skipCache and desc in self.brush_cache:
            # print('using cached brush', desc)
            return self.brush_cache[desc]
        # print('making brush!')
        qbrush = QtGui.QBrush(QtCore.Qt.SolidPattern) # make sure this brush fills once color is set
        self._update_QBrush(qbrush,desc)
        if register: self.register( qbrush, desc ) # register for updates on palette change
        if not skipCache: # skipCache disable both reading and writing cache
            self.brush_cache[desc] = qbrush
        return qbrush
        
    def _update_QColor(self, qcol, desc):
        """ updates qcol to match descriptor for current palette """
        # print('updating color to match',desc)
        name, alpha = desc
        if name[0] != '#':
            qcol.setRgba( self.color_dic[name].rgba() )
        else:
            qcol.setNamedColor(name) # set from hex string
        if alpha is not None: qcol.setAlpha(alpha)
        # print('= hex:', qcol.name() )

    def _update_QPen(self, qpen, desc):
        """ updates qpen to match descriptor for current palette """
        # print('updating pen to match',desc)
        name, width, alpha = desc
        if name[0] != '#':
            if alpha is None:
                qpen.setColor( self.color_dic[name] ) # automatically copies
            else:
                qcol = QtGui.QColor(self.color_dic[name]) # make a copy
                qcol.setAlpha(alpha)
                qpen.setColor(qcol)
        else: # set from hex string instead:
            qcol = QtGui.QColor(name)
            if alpha is not None: qcol.setAlpha(alpha)
            qpen.setColor(qcol)
        if width is not None:
            qpen.setWidth(width)
        # print('= hex:', qpen.color().name() )

    def _update_QBrush(self, qbrush, desc):
        """ updates qbrush to match descriptor for current palette """
        # print('updating brush',qbrush,'to match',desc)
        name, alpha = desc
        if name[0] != '#':
            if alpha is None:
                qbrush.setColor( self.color_dic[name] ) # automatically copies
            else:
                qcol = QtGui.QColor(self.color_dic[name]) # make a copy
                qcol.setAlpha(alpha)
                qbrush.setColor(qcol)
        else: # set from hex string instead:
            qcol = QtGui.QColor(name)
            if alpha is not None: qcol.setAlpha(alpha)
            qbrush.setColor(qcol)
        # print('= hex:', qbrush.color().name(), qbrush.color().alpha() )
        

    def register(self, obj, desc):
        """
        Registers obj (QColor, QPen or QBrush) to be updated according to the descriptor on palette change
        """
        if hasattr(obj,'registration'):
            registration = obj.registration
        else:
            registration = next(ColorRegistry._registrationGenerator)
            obj.registration = registration # patch in attribute
        fin = weakref.finalize(obj, self.unregister, registration)
        fin.atexit = False # no need to clean up registry on program exit
        # print('registering', registration, '(',str(obj),'):',str(desc))
        self.registered_objects[registration] = (weakref.ref(obj), desc)

    def unregister(self, registration):
        """
        Removes obj (QColor, QPen or QBrush) from the registry, usually called by finalize on deletion
        """
        obj, desc = self.registered_objects[registration]
        # print('unregistering', registration, '(',str(obj),'):',str(desc))
        del self.registered_objects[registration]
        del obj, desc


    def colors(self):
        """ return current list of colors """
        return self.color_dic # it would be safer (but slower) to provide only a copy
    

    def redefinePalette(self, colors=None):
        """ 
        Update list of named colors if 'colors' dictionary is given
        Emits paletteHasChanged signals to color objects and widgets, even if color_dict is None 
        """
        if colors is not None:
            for key in DEFAULT_COLORS:
                if key not in colors:
                    raise ValueError("Palette definition is missing '"+str(key)+"'")
            if DEBUG: print('  ColorRegistry: Setting palette, all color definitions are present.')
            self.color_dic.clear()
            self.color_dic.update(colors)

        # notifies named color objects of new assignments:
        for key in self.registered_objects:
            ref, desc = self.registered_objects[key]
            obj = ref()
            # print('updating', obj)
            if obj is None:
                warnings.warn('Expired object with descriptor '+str(desc)+' remains in color registry.', RuntimeWarning)
            elif isinstance(obj, QtGui.QColor):
                self._update_QColor(obj, desc)
            elif isinstance(obj, QtGui.QPen):
                self._update_QPen(obj, desc)
            elif isinstance(obj, QtGui.QBrush):
                self._update_QBrush(obj, desc)
        # notify all graphics widgets that redraw is required:
        self.paletteHasChangedSignal.emit()
