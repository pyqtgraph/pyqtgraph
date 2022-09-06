from collections import OrderedDict

import numpy as np

from .. import functions as fn
from .. import parametertree as ptree
from ..Qt import QtCore

__all__ = ['DataFilterWidget']


class DataFilterWidget(ptree.ParameterTree):
    """
    This class allows the user to filter multi-column data sets by specifying
    multiple criteria
    
    Wraps methods from DataFilterParameter: setFields, generateMask,
    filterData, and describe.
    """
    
    sigFilterChanged = QtCore.Signal(object)
    
    def __init__(self):
        ptree.ParameterTree.__init__(self, showHeader=False)
        self.params = DataFilterParameter()
        
        self.setParameters(self.params)
        self.params.sigFilterChanged.connect(self.sigFilterChanged)
        
        self.setFields = self.params.setFields
        self.generateMask = self.params.generateMask
        self.filterData = self.params.filterData
        self.describe = self.params.describe
        
    def parameters(self):
        return self.params

    def addFilter(self, name):
        """Add a new filter and return the created parameter item.
        """
        return self.params.addNew(name)

        
class DataFilterParameter(ptree.types.GroupParameter):
    """A parameter group that specifies a set of filters to apply to tabular data.
    """
    sigFilterChanged = QtCore.Signal(object)
    
    def __init__(self):
        self.fields = {}
        ptree.types.GroupParameter.__init__(self, name='Data Filter', addText='Add filter..', addList=[])
        self.sigTreeStateChanged.connect(self.filterChanged)
    
    def filterChanged(self):
        self.sigFilterChanged.emit(self)
        
    def addNew(self, name):
        mode = self.fields[name].get('mode', 'range')
        if mode == 'range':
            child = self.addChild(RangeFilterItem(name, self.fields[name]))
        elif mode == 'enum':
            child = self.addChild(EnumFilterItem(name, self.fields[name]))
        else:
            raise ValueError("field mode must be 'range' or 'enum'")
        return child
            
    def fieldNames(self):
        return self.fields.keys()
    
    def setFields(self, fields):
        """Set the list of fields that are available to be filtered.

        *fields* must be a dict or list of tuples that maps field names
        to a specification describing the field. Each specification is
        itself a dict with either ``'mode':'range'`` or ``'mode':'enum'``::

            filter.setFields([
                ('field1', {'mode': 'range'}),
                ('field2', {'mode': 'enum', 'values': ['val1', 'val2', 'val3']}),
                ('field3', {'mode': 'enum', 'values': {'val1':True, 'val2':False, 'val3':True}}),
            ])
        """
        with fn.SignalBlock(self.sigTreeStateChanged, self.filterChanged):
            self.fields = OrderedDict(fields)
            names = self.fieldNames()
            self.setAddList(names)

            # update any existing filters
            for ch in self.children():
                name = ch.fieldName
                if name in fields:
                    ch.updateFilter(fields[name])
        self.sigFilterChanged.emit(self)
    
    def filterData(self, data):
        if len(data) == 0:
            return data
        return data[self.generateMask(data)]
    
    def generateMask(self, data):
        """Return a boolean mask indicating whether each item in *data* passes
        the filter critera.
        """
        mask = np.ones(len(data), dtype=bool)
        if len(data) == 0:
            return mask
        for fp in self:
            if fp.value() is False:
                continue
            mask &= fp.generateMask(data, mask.copy())
            #key, mn, mx = fp.fieldName, fp['Min'], fp['Max']
            
            #vals = data[key]
            #mask &= (vals >= mn)
            #mask &= (vals < mx)  ## Use inclusive minimum and non-inclusive maximum. This makes it easier to create non-overlapping selections
        return mask
    
    def describe(self):
        """Return a list of strings describing the currently enabled filters."""
        desc = []
        for fp in self:
            if fp.value() is False:
                continue
            desc.append(fp.describe())
        return desc


class RangeFilterItem(ptree.types.SimpleParameter):
    def __init__(self, name, opts):
        self.fieldName = name
        units = opts.get('units', '')
        self.units = units
        ptree.types.SimpleParameter.__init__(self, 
            name=name, autoIncrementName=True, type='bool', value=True, removable=True, renamable=True, 
            children=[
                #dict(name="Field", type='list', value=name, limits=fields),
                dict(name='Min', type='float', value=0.0, suffix=units, siPrefix=True),
                dict(name='Max', type='float', value=1.0, suffix=units, siPrefix=True),
            ])
            
    def generateMask(self, data, mask):
        vals = data[self.fieldName][mask]
        mask[mask] = (vals >= self['Min']) & (vals < self['Max'])  ## Use inclusive minimum and non-inclusive maximum. This makes it easier to create non-overlapping selections
        return mask
    
    def describe(self):
        return "%s < %s < %s" % (fn.siFormat(self['Min'], suffix=self.units), self.fieldName, fn.siFormat(self['Max'], suffix=self.units))

    def updateFilter(self, opts):
        pass
    

class EnumFilterItem(ptree.types.SimpleParameter):
    def __init__(self, name, opts):
        self.fieldName = name
        ptree.types.SimpleParameter.__init__(self, 
            name=name, autoIncrementName=True, type='bool', value=True, removable=True, renamable=True)
        self.setEnumVals(opts)            
    
    def generateMask(self, data, startMask):
        vals = data[self.fieldName][startMask]
        mask = np.ones(len(vals), dtype=bool)
        otherMask = np.ones(len(vals), dtype=bool)
        for c in self:
            key = c.maskValue
            if key == '__other__':
                m = ~otherMask
            else:
                m = vals != key
                otherMask &= m
            if c.value() is False:
                mask &= m
        startMask[startMask] = mask
        return startMask

    def describe(self):
        vals = [ch.name() for ch in self if ch.value() is True]
        return "%s: %s" % (self.fieldName, ', '.join(vals))

    def updateFilter(self, opts):
        self.setEnumVals(opts)

    def setEnumVals(self, opts):
        vals = opts.get('values', {})

        prevState = {}
        for ch in self.children():
            prevState[ch.name()] = ch.value()
            self.removeChild(ch)

        if not isinstance(vals, dict):
            vals = OrderedDict([(v,(str(v), True)) for v in vals])
        
        # Each filterable value can come with either (1) a string name, (2) a bool
        # indicating whether the value is enabled by default, or (3) a tuple providing
        # both.
        for val,valopts in vals.items():
            if isinstance(valopts, bool):
                enabled = valopts
                vname = str(val)
            elif isinstance(valopts, str):
                enabled = True
                vname = valopts
            elif isinstance(valopts, tuple):
                vname, enabled = valopts

            ch = ptree.Parameter.create(name=vname, type='bool', value=prevState.get(vname, enabled))
            ch.maskValue = val
            self.addChild(ch)
        ch = ptree.Parameter.create(name='(other)', type='bool', value=prevState.get('(other)', True))
        ch.maskValue = '__other__'
        self.addChild(ch)
