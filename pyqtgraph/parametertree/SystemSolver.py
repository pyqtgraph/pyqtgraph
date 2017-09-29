from ..pgcollections import OrderedDict
import numpy as np
import copy


class SystemSolver(object):
    """
    This abstract class is used to formalize and manage user interaction with a 
    complex system of equations (related to "constraint satisfaction problems").
    It is often the case that devices must be controlled
    through a large number of free variables, and interactions between these 
    variables make the system difficult to manage and conceptualize as a user
    interface. This class does _not_ attempt to numerically solve the system
    of equations. Rather, it provides a framework for subdividing the system
    into manageable pieces and specifying closed-form solutions to these small 
    pieces.
    
    For an example, see the simple Camera class below.
    
    Theory of operation: Conceptualize the system as 1) a set of variables
    whose values may be either user-specified or automatically generated, and 
    2) a set of functions that define *how* each variable should be generated. 
    When a variable is accessed (as an instance attribute), the solver first
    checks to see if it already has a value (either user-supplied, or cached
    from a previous calculation). If it does not, then the solver calls a 
    method on itself (the method must be named `_variableName`) that will
    either return the calculated value (which usually involves acccessing
    other variables in the system), or raise RuntimeError if it is unable to
    calculate the value (usually because the user has not provided sufficient
    input to fully constrain the system). 
    
    Each method that calculates a variable value may include multiple 
    try/except blocks, so that if one method generates a RuntimeError, it may 
    fall back on others. 
    In this way, the system may be solved by recursively searching the tree of 
    possible relationships between variables. This allows the user flexibility
    in deciding which variables are the most important to specify, while 
    avoiding the apparent combinatorial explosion of calculation pathways
    that must be considered by the developer.
    
    Solved values are cached for efficiency, and automatically cleared when 
    a state change invalidates the cache. The rules for this are simple: any
    time a value is set, it invalidates the cache *unless* the previous value
    was None (which indicates that no other variable has yet requested that 
    value). More complex cache management may be defined in subclasses.
    
    
    Subclasses must define:
    
    1) The *defaultState* class attribute: This is a dict containing a 
       description of the variables in the system--their default values,
       data types, and the ways they can be constrained. The format is::
       
           { name: [value, type, constraint, allowed_constraints], ...}
       
       * *value* is the default value. May be None if it has not been specified
         yet.
       * *type* may be float, int, bool, np.ndarray, ...
       * *constraint* may be None, single value, or (min, max)
            * None indicates that the value is not constrained--it may be 
              automatically generated if the value is requested.
       * *allowed_constraints* is a string composed of (n)one, (f)ixed, and (r)ange. 
       
       Note: do not put mutable objects inside defaultState!
       
    2) For each variable that may be automatically determined, a method must 
       be defined with the name `_variableName`. This method may either return
       the 
    """

    defaultState = OrderedDict()

    def __init__(self):
        self.__dict__['_vars'] = OrderedDict()
        self.__dict__['_currentGets'] = set()
        self.reset()
        
    def copy(self):
        sys = type(self)()
        sys.__dict__['_vars'] = copy.deepcopy(self.__dict__['_vars'])
        sys.__dict__['_currentGets'] = copy.deepcopy(self.__dict__['_currentGets'])
        return sys

    def reset(self):
        """
        Reset all variables in the solver to their default state.
        """
        self._currentGets.clear()
        for k in self.defaultState:
            self._vars[k] = self.defaultState[k][:]

    def __getattr__(self, name):
        if name in self._vars:
            return self.get(name)
        raise AttributeError(name)
    
    def __setattr__(self, name, value):
        """
        Set the value of a state variable. 
        If None is given for the value, then the constraint will also be set to None.
        If a tuple is given for a scalar variable, then the tuple is used as a range constraint instead of a value.
        Otherwise, the constraint is set to 'fixed'.
        
        """
        # First check this is a valid attribute
        if name in self._vars:
            if value is None:
                self.set(name, value, None)
            elif isinstance(value, tuple) and self._vars[name][1] is not np.ndarray:
                self.set(name, None, value)
            else:
                self.set(name, value, 'fixed')
        else:
            # also allow setting any other pre-existing attribute
            if hasattr(self, name):
                object.__setattr__(self, name, value)
            else:
                raise AttributeError(name)
        
    def get(self, name):
        """
        Return the value for parameter *name*. 
        
        If the value has not been specified, then attempt to compute it from
        other interacting parameters.
        
        If no value can be determined, then raise RuntimeError.
        """
        if name in self._currentGets:
                raise RuntimeError("Cyclic dependency while calculating '%s'." % name)
        self._currentGets.add(name)
        try:
            v = self._vars[name][0]
            if v is None:
                cfunc = getattr(self, '_' + name, None)
                if cfunc is None:
                    v = None
                else:
                    v = cfunc()
                if v is None:
                    raise RuntimeError("Parameter '%s' is not specified." % name)
                v = self.set(name, v)
        finally:
            self._currentGets.remove(name)
        
        return v
    
    def set(self, name, value=None, constraint=True):
        """
        Set a variable *name* to *value*. The actual set value is returned (in
        some cases, the value may be cast into another type).
        
        If *value* is None, then the value is left to be determined in the 
        future. At any time, the value may be re-assigned arbitrarily unless
        a constraint is given.
        
        If *constraint* is True (the default), then supplying a value that 
        violates a previously specified constraint will raise an exception.
        
        If *constraint* is 'fixed', then the value is set (if provided) and
        the variable will not be updated automatically in the future.

        If *constraint* is a tuple, then the value is constrained to be within the 
        given (min, max). Either constraint may be None to disable 
        it. In some cases, a constraint cannot be satisfied automatically,
        and the user will be forced to resolve the constraint manually.
        
        If *constraint* is None, then any constraints are removed for the variable.
        """
        var = self._vars[name]
        if constraint is None:
            if 'n' not in var[3]:
                raise TypeError("Empty constraints not allowed for '%s'" % name)
            var[2] = constraint
        elif constraint == 'fixed':
            if 'f' not in var[3]:
                raise TypeError("Fixed constraints not allowed for '%s'" % name)
            var[2] = constraint
        elif isinstance(constraint, tuple):
            if 'r' not in var[3]:
                raise TypeError("Range constraints not allowed for '%s'" % name)
            assert len(constraint) == 2
            var[2] = constraint
        elif constraint is not True:
            raise TypeError("constraint must be None, True, 'fixed', or tuple. (got %s)" % constraint)
        
        # type checking / massaging
        if var[1] is np.ndarray and value is not None:
            value = np.array(value, dtype=float)
        elif var[1] in (int, float, tuple) and value is not None:
            value = var[1](value)
            
        # constraint checks
        if constraint is True and not self.check_constraint(name, value):
            raise ValueError("Setting %s = %s violates constraint %s" % (name, value, var[2]))

        # invalidate other dependent values
        if var[0] is not None or value is None:
            # todo: we can make this more clever..(and might need to) 
            # we just know that a value of None cannot have dependencies
            # (because if anyone else had asked for this value, it wouldn't be 
            # None anymore)
            self.resetUnfixed()
            
        var[0] = value
        return value
    
    def check_constraint(self, name, value):
        c = self._vars[name][2]
        if c is None or value is None:
            return True
        if isinstance(c, tuple):
            return ((c[0] is None or c[0] <= value) and
                    (c[1] is None or c[1] >= value))
        else:
            return value == c
    
    def saveState(self):
        """
        Return a serializable description of the solver's current state.
        """
        state = OrderedDict()
        for name, var in self._vars.items():
            state[name] = (var[0], var[2])
        return state
    
    def restoreState(self, state):
        """
        Restore the state of all values and constraints in the solver.
        """
        self.reset()
        for name, var in state.items():
            self.set(name, var[0], var[1])
    
    def resetUnfixed(self):
        """
        For any variable that does not have a fixed value, reset
        its value to None.
        """
        for var in self._vars.values():
            if var[2] != 'fixed':
                var[0] = None
                
    def solve(self):
        for k in self._vars:
            getattr(self, k)
                
    def __repr__(self):
        state = OrderedDict()
        for name, var in self._vars.items():
            if var[2] == 'fixed':
                state[name] = var[0]
        state = ', '.join(["%s=%s" % (n, v) for n,v in state.items()])
        return "<%s %s>" % (self.__class__.__name__, state)





if __name__ == '__main__':
    
    class Camera(SystemSolver):
        """
        Consider a simple SLR camera. The variables we will consider that 
        affect the camera's behavior while acquiring a photo are aperture, shutter speed,
        ISO, and flash (of course there are many more, but let's keep the example simple).

        In rare cases, the user wants to manually specify each of these variables and
        no more work needs to be done to take the photo. More often, the user wants to
        specify more interesting constraints like depth of field, overall exposure, 
        or maximum allowed ISO value.

        If we add a simple light meter measurement into this system and an 'exposure'
        variable that indicates the desired exposure (0 is "perfect", -1 is one stop 
        darker, etc), then the system of equations governing the camera behavior would
        have the following variables:

            aperture, shutter, iso, flash, exposure, light meter

        The first four variables are the "outputs" of the system (they directly drive 
        the camera), the last is a constant (the camera itself cannot affect the 
        reading on the light meter), and 'exposure' specifies a desired relationship 
        between other variables in the system.

        So the question is: how can I formalize a system like this as a user interface?
        Typical cameras have a fairly limited approach: provide the user with a list
        of modes, each of which defines a particular set of constraints. For example:

            manual: user provides aperture, shutter, iso, and flash
            aperture priority: user provides aperture and exposure, camera selects
                            iso, shutter, and flash automatically
            shutter priority: user provides shutter and exposure, camera selects
                            iso, aperture, and flash
            program: user specifies exposure, camera selects all other variables
                    automatically
            action: camera selects all variables while attempting to maximize 
                    shutter speed
            portrait: camera selects all variables while attempting to minimize 
                    aperture

        A more general approach might allow the user to provide more explicit 
        constraints on each variable (for example: I want a shutter speed of 1/30 or 
        slower, an ISO no greater than 400, an exposure between -1 and 1, and the 
        smallest aperture possible given all other constraints) and have the camera 
        solve the system of equations, with a warning if no solution is found. This
        is exactly what we will implement in this example class.
        """

        defaultState = OrderedDict([
            # Field stop aperture
            ('aperture', [None, float, None, 'nf']),
            # Duration that shutter is held open. 
            ('shutter', [None, float, None, 'nf']),
            # ISO (sensitivity) value. 100, 200, 400, 800, 1600.. 
            ('iso', [None, int, None, 'nf']),
            
            # Flash is a value indicating the brightness of the flash. A table
            # is used to decide on "balanced" settings for each flash level:
            #   0: no flash
            #   1: s=1/60,  a=2.0, iso=100
            #   2: s=1/60,  a=4.0, iso=100   ..and so on..
            ('flash', [None, float, None, 'nf']),
            
            # exposure is a value indicating how many stops brighter (+1) or
            # darker (-1) the photographer would like the photo to appear from
            # the 'balanced' settings indicated by the light meter (see below).
            ('exposure', [None, float, None, 'f']),
            
            # Let's define this as an external light meter (not affected by 
            # aperture) with logarithmic output. We arbitrarily choose the
            # following settings as "well balanced" for each light meter value:
            #   -1: s=1/60,  a=2.0, iso=100
            #    0: s=1/60,  a=4.0, iso=100
            #    1: s=1/120, a=4.0, iso=100    ..and so on..
            # Note that the only allowed constraint mode is (f)ixed, since the
            # camera never _computes_ the light meter value, it only reads it.
            ('lightMeter', [None, float, None, 'f']),  
            
            # Indicates the camera's final decision on how it thinks the photo will 
            # look, given the chosen settings. This value is _only_ determined
            # automatically.
            ('balance', [None, float, None, 'n']),
            ])
        
        def _aperture(self):
            """
            Determine aperture automatically under a variety of conditions.
            """
            iso = self.iso
            exp = self.exposure
            light = self.lightMeter
            
            try:
                # shutter-priority mode
                sh = self.shutter   # this raises RuntimeError if shutter has not
                                   # been specified
                ap = 4.0 * (sh / (1./60.)) * (iso / 100.) * (2 ** exp) * (2 ** light)
                ap = np.clip(ap, 2.0, 16.0)
            except RuntimeError:
                # program mode; we can select a suitable shutter
                # value at the same time.
                sh = (1./60.)
                raise
            
            
            
            return ap

        def _balance(self):
            iso = self.iso
            light = self.lightMeter
            sh = self.shutter
            ap = self.aperture
            fl = self.flash
            
            bal = (4.0 / ap) * (sh / (1./60.)) * (iso / 100.) * (2 ** light)
            return np.log2(bal)
    
    camera = Camera()
    
    camera.iso = 100
    camera.exposure = 0
    camera.lightMeter = 2
    camera.shutter = 1./60.
    camera.flash = 0
    
    camera.solve()
    print(camera.saveState())
    