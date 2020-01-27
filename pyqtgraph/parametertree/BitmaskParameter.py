# -*- coding: utf-8 -*-
"""
A pyqtgraph.parametertree.Parameter subclass to represent a bitmask,
with the 'bitmask' type (BitmaskParameter) for handlig enum.IntFlag,
available from Python 3.6 https://docs.python.org/3/library/enum.html#enum.IntFlag
"""

from . import parameterTypes
from .Parameter import registerParameterType
from .ParameterItem import ParameterItem
from enum import IntFlag

class BitmaskParameterItem(parameterTypes.GroupParameterItem):
    def __init__(self, param, depth=0):
        super().__init__(param, depth)
        param.sigValueChanged.connect(self.valueChanged)
        f = self.font(1)
        f.setBold(False)
        self.setFont(1, f)
        self.valueChanged(param, param.value())
        
    def valueChanged(self, param, val):
        if param.opts['numeric']:
            # Show the numeric value to the right of the name (title)
            val = int(val)
            self.setText(1, format('{} = 0b{:b}'.format(val, val))) # decimal and binary form
        
    def treeWidgetChanged(self):
        if self.param.opts['numeric']:
            # Avoid the GroupParameterItem's implementation.
            # A colspan of 2 for name would not allow showing the numeric value.
            ParameterItem.treeWidgetChanged(self)
        else:
            # OK to let the name (title) span two columns
            # when we don't show the numeric value.
            super().treeWidgetChanged()
        
class BitmaskParameter(parameterTypes.GroupParameter):
    """A pyqtgraph.parametertree.Parameter subclass to represent a bitmask,
    e.g. of type enum.IntFlag, by showing one boolean sub-parameter (checkbox)
    per bit in the bitmask. IntFlag is available from Python 3.6.
    
    Example:
    from enum import IntFlag
    from pyqtgraph.parametertree.BitmaskParameter import BitmaskParameter
    # this import registers 'bitmask' as a Parameter type
    
    class ChannelMask(IntFlag):
        '''Represents a choice of zero, one or multiple channels.'''
        A = 0b0001 # only the first channel
        B = 0b0010 # only the second channel
        C = 0b0100 # only the third channel
        D = 0b1000 # only the fourth channel
    
    # Different forms of explicit construction of a BitmaskParameter
    structure = [{'type': 'group', 'name': 'Examples', 'children': [
            {'type': 'float', 'name': 'A number', 'value': 1.23},
            BitmaskParameter(name='Example 1', value=ChannelMask.A|ChannelMask.B),
            BitmaskParameter(name='Example 2', value=ChannelMask.A|ChannelMask.B,
                             numeric=False), # skip the numeric indicator
            BitmaskParameter(name='Example 3', value=3, values=dict(ChannelMask.__members__)),
                             # allows value to be a plain integer
            BitmaskParameter(name='Empty start', values=dict(ChannelMask.__members__)), 
                             # will have 0 as default value
            BitmaskParameter(name='Restricted', 
                             values={f.name: f.value for f in list(ChannelMask)[0:2]}), 
                             # show only the first two bits
            {'type': 'str', 'name': 'Something else', 'value': ''}
        ]}]
    
    # Indirect construction from dict
    structure = [{'type': 'group', 'name': 'Examples', 'children': [
            {'type': 'float', 'name': 'A number', 'value': 1.23},
            {'type': 'bitmask', 'name': 'Channels', 'value': ChannelMask.A|ChannelMask.B},
            {'type': 'bitmask', 'name': 'Channels without showing integer', 
                 'numeric': False, # don't show the numeric indicator
                 'value': 3, 'values': dict(ChannelMask.__members__)
                 }, # accepts plain integer value (since 'values' is given)
            {'type': 'bitmask', 'name': 'Empty start', 'values': dict(ChannelMask.__members__)},
            {'type': 'bitmask', 'name': 'Restricted', 
                 'values': {f.name: f.value for f in list(ChannelMask)[0:2]}},
                 # show only the first two bits
            {'type': 'str', 'name': 'Something else', 'value': ''}
        ]}]
    
    ## Create tree of Parameter objects
    paramTree = Parameter.create(name='root', type='group', children=structure)
    """
    
    itemClass = BitmaskParameterItem
    
    def __init__(self, **opts):
        if 'addText' in opts:
            # restrict the GroupParameter implementation by not allowing buttons
            opts.pop('addText')
        if 'numeric' not in opts:
            opts['numeric'] = True # default to True
        opts['type'] = 'int'
        if 'values' in opts:
            values = opts['values']
            if not isinstance(values, dict):
                raise ValueError('The BitmaskParameter expects a '
                                 'dict of name: flag-value mappings.')
            if not 'value' in opts:
                # If no initial value is given, start with all checkboxed unchecked.
                if isinstance(list(opts['values'].values())[0], IntFlag):
                    # Represent the 0-value with an object 
                    # of the same IntFlag-subclass as other values
                    opts['value'] = type(list(values.values())[0])(0)
                else: 
                    # Fall back to the plain integer 0
                    opts['value'] = 0
                # Alternative in case we'd rather use the first of the listed values:
                #opts['value'] = list(values.values())[0] # use the first listed value
        else:
            if not 'value' in opts:
                raise ValueError("Neither an initial value nor a list of "
                         "possible values was given for a BitmaskParameter parameter.")
            elif isinstance(opts['value'], IntFlag):
                # If the value is an IntFlag we can automatically get the list
                # of named flag-bits
                opts['values'] = dict(type(opts['value']).__members__)
            else:
                raise ValueError("Neither an initial value nor a list of possible "
                         "flag-values was given for a BitmaskParameter parameter.")
        
        self._parametersAndFlags = [] # (SimpleParameter, IntFlag)-tuples
        parameterTypes.GroupParameter.__init__(self, **opts)
        for name, flag in opts['values'].items():
            # Create a boolean parameter child per flag-bit in the mask
            self.addChild({'name': name, 'type': 'bool', 
                           'value': bool(flag & opts['value'])})
            p = self.child(name) # get the created SimpleParameter instance
            self._parametersAndFlags.append((p, flag))
            
            # disconnect all slots so it doesn't directly trigger onConfTreeChange
            p.sigValueChanged.disconnect() 
            # inform this instance when sub-parameter checkbox is checked/unchecked by user
            p.sigValueChanged.connect(self.bitChanged) 
            # (p.sigChanging won't be emitted for boolean parameters)
        self.setValue(opts['value']) # repeat now that _parametersAndFlags is initialized
        
    def _interpretValue(self, v):
        """This is called by Parameter.setValue() to convert an incoming value
        to a valid value. The type of the value is preserved,
        assuming it behaves as an integer (which IntFlag does).
        """
        if int(v) != v:
            # Unexpected type. Use whatever int() gives (if not raising ValueError)
            v = int(v)
        return v
        
    def setValue(self, value, blockSignal=None):
        """Parameter.setValue() writes the value to self.opts['value']
        and emits signals.
        """
        super().setValue(value, blockSignal)
        # Here we need to update the single-bit-parameters
        # (shown using WidgetParameterItem for 'bool')
        for p, f in self._parametersAndFlags:
            # Update the checkbox but block it from signalling back
            # to bitChanged() to avoid a signal loop
            p.setValue(bool(f & self.opts['value']), self.bitChanged)
        
    def bitChanged(self):
        """Computes a new value for the mask when the user checks/unchecks
        any checkbox.
        """
        value = 0
        first = True
        # Here we need to update the single-bit-parameters (shown using 
        # WidgetParameterItem for 'bool')
        for p, f in self._parametersAndFlags:
            if first:
                # Convert the default 0 to the correct type (as when some 
                # checkbox is checked), to not let data type vary with whether 
                # any checkbox is checked or not.
                value = f & 0
                first = False
            if p.value():
                # Enable the bit corresponding to this boolean parameter
                value = value | f
        # Call setValue() which will call _interpretValue(),
        # update self.opt['value'] and then signal the change to any listeners.
        super().setValue(value)
        
registerParameterType('bitmask', BitmaskParameter, override=True)