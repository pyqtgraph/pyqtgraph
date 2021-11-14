import warnings
import numpy as np

__all__ = []

MAPPINGS = {}

def get(name):
    """ 
    Returns PlotDataMapping according to specified name
    """
    if name not in MAPPINGS:
        raise KeyError(f"No PlotDataMapping '{name}' has been registered.")
    return MAPPINGS[name]
    
def register(name, mapping):
    """
    Registers a new PlotDataMapping object
    """
    if not isinstance(mapping, PlotDataMapping):
        raise TypeError('added object must subclass PlotDataMapping')
    MAPPINGS[name] = mapping
    

class PlotDataMapping(object):
    """
    :orphan:
    .. warning:: This class is still experimental and the interface may change without warning.
    
    Maintains all required information related to a mapping that is applied to 
    data (typically by `pyqtgraph.PlotDataItem`) before it is rendered to screen.
    
    Different mapping functions are implemented by 
    
    Public properties
    -----------------
    vsLimits: tuple (min, max)
        The maximum range allowed for ViewBox zoom
    """
    def __init__(self, name):
        """ initialize and look up limits """
        self.name = name
        # self.vsLimits = (None, None) # subclasses need to implement this.
        
    def __str__(self):
        return f"{self.name} mapping"

    def map(self, dsValues, finiteCheck=False):
        """ 
        Maps values from data space to view space.
        Returns `mappedData`
        
        If `finiteCheck = True`, the data is also checked for non-finite values.
        Returns `(mappedData, containsNonFinite)`
        If `containsNonFinite == None`, no information is available.
        """
        del dsValues
        if not finiteCheck:
            return None
        # also report non-finites:
        return None, None

    def mapFloat(self, vsValue):
        """
        Maps single float value from data space to view space.
        Returns `mappedValue`
        """
        return None

    def reverse(self, vsValues, finiteCheck=False):
        """
        Reverse maps values from view space to data space.
        Returns `reversedData`
        
        If `finiteCheck = True`, the data is also checked for non-finite values.
        Returns `(reversedData, containsNonFinite)`
        If `containsNonFinite == None`, no information is available.
        """
        del vsValues
        if not finiteCheck:
            return None
        # also report non-finites:
        return None, None
        
    def reverseFloat(self, vsValue):
        """
        Reverse maps single float value from view space to data space.
        Returns `reversedValue`
        """
        return None


class IdentityMapping(PlotDataMapping):
    """
    :orphan:
    .. warning:: This class is still experimental and the interface may change without warning.
    Identity mapping That does not affect the data, but still provides the required limits and interface.
    """
    def __init__(self):
        super().__init__('identity')
        info = np.finfo(float)
        # If we let each axis go to the full range, then the span cannot be calculated
        # If we limit to min/2 to max/2, scaling calculations still fail
        self.vsLimits = (info.min/4, info.max/4)

    def map(self, dsValues, finiteCheck=False):
        vsValues = dsValues # no mapping is applied
        if not finiteCheck:
            return vsValues
        # also report non-finites:
        return vsValues, None 

    def mapFloat(self, dsValue):
        return dsValue

    def reverse(self, vsValues, finiteCheck=False):
        dsValues = vsValues
        if not finiteCheck:
            return dsValues
        # also report non-finites:
        return dsValues, None 
        
    def reverseFloat(self, vsValue):
        return vsValue


class LogMapping(PlotDataMapping):
    """
    :orphan:
    .. warning:: This class is still experimental and the interface may change without warning.
    Conventional base-10 logarithmic mapping
    """    
    def __init__(self):
        super().__init__('log')
        self.vsLimits = (
            -307.6, # the smallest absolute value that can be represented is 2.2E-308
            np.log10( np.finfo(float).max )
        )

    def map(self, dsValues, finiteCheck=False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vsValues = np.log10(dsValues)
        nonfinites = ~np.isfinite( vsValues )
        if nonfinites.any():
            vsValues[nonfinites] = np.nan # set all non-finite values to NaN
            containsNonfinite = True
        else:
            containsNonfinite = False
        if not finiteCheck:
            return vsValues
        # also report non-finites:
        return vsValues, containsNonfinite 
        
    def mapFloat(self, dsValue):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vsValue = np.log10( float(dsValue) )
        if not np.isfinite(vsValue): vsValue = np.nan
        return vsValue            

    def reverse(self, vsValues, finiteCheck=False):
        dsValues = 10**vsValues # np.nan silently results in np.nan
        if not finiteCheck:
            return dsValues
        # also report non-finites:
        nonfinites = ~np.isfinite( dsValues )
        if nonfinites.any():
            containsNonfinite = True
        else:
            containsNonfinite = False
        return dsValues, containsNonfinite 

    def reverseFloat(self, vsValue):
        if vsValue < self.vsLimits[0]: return np.nan
        if vsValue > self.vsLimits[1]: return np.nan
        dsValue = 10**float(vsValue)
        return dsValue
        
        
register('identity', IdentityMapping() )
register('log'     , LogMapping() )
