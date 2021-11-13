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
        self.vsLimits = (None, None) 

    def map(self, dsValues):
        """ 
        Maps values from data space to view space.

        Returns `(mappedData, containsNonFinite)`
        If `containsNonFinite == None`, no information is available.
        """
        del dsValues
        return None, None
    
    def reverse(self, vsValues):
        """
        Reverse maps values from view space to data space.
        
        Returns `(mappedData, containsNonFinite)`
        If `containsNonFinite == None`, no information is available.
        """
        del vsValues
        return None, None
        dsValues = vsValues
        return dsValues


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

    def map(self, dsValues):
        vsValues = dsValues # no mapping is applied
        return vsValues, None 

    def reverse(self, vsValues):
        dsValues = vsValues
        return dsValues, None


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

    def map(self, dsValues):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vsValues = np.log10(dsValues)
        nonfinites = ~np.isfinite( vsValues )
        if nonfinites.any():
            vsValues[nonfinites] = np.nan # set all non-finite values to NaN
            containsNonfinite = True
        else:
            containsNonfinite = False
        return vsValues, containsNonfinite 
    
    def reverse(self, vsValues):
        dsValues = 10**vsValues # np.nan silently results in np.nan
        return dsValues, None
        
register('identity', IdentityMapping() )
register('log'     , LogMapping() )
