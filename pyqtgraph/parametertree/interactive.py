import contextlib
import functools
import inspect
import pydoc
import weakref

from . import Parameter


class RunOpts:
    class PARAM_UNSET:
        pass

    """Sentinel value for detecting parameters with unset values"""

    ON_BUTTON = "button"
    """Indicator for ``interactive`` parameter which runs the function on pressing a button parameter"""
    ON_CHANGED = "changed"
    """
    Indicator for ``interactive`` parameter which runs the function every time one ``sigValueChanged`` is emitted from
    any of the parameters
    """
    ON_CHANGING = "changing"
    """
    Indicator for ``interactive`` parameter which runs the function every time one ``sigValueChanging`` is emitted from
    any of the parameters
    """


class _InteractDefaults:
    def __init__(self):
        self.runOpts = RunOpts.ON_CHANGED
        self.parent = None
        self.title = None
        self.nest = True
        self.existOk = True

        self.runButtonTemplate = dict(type="action", defaultName="Run")

    def setOpts(self, **opts):
        oldOpts = vars(self).copy()
        allowed = set(oldOpts)
        errors = set(opts).difference(allowed)
        if errors:
            raise KeyError(f"Unrecognized options: {errors}. Must be one of: {allowed}")

        toReturn = {}
        toUse = {}
        for kk in allowed.intersection(opts):
            toReturn[kk] = oldOpts[kk]
            toUse[kk] = opts[kk]
        self.__dict__.update(toUse)
        return toReturn

    @contextlib.contextmanager
    def optsContext(self, **opts):
        oldOpts = self.__dict__.copy()
        self.setOpts(**opts)
        yield
        self.setOpts(**oldOpts)

    def __str__(self):
        return f'"interact" defaults: {vars(self)}'

    def __repr__(self):
        return str(self)


interactDefaults = _InteractDefaults()


def setRunButtonTemplate(template):
    """
    Changes the default template for buttons created when "ON_BUTTON" run behavior is requested.
    A ``Parameter` is created from this dict that must function like an ActionParameter.
    a ``defaultName`` key determines the button's name if ``nest=False`` during creation.
    The old template is returned for convenient resetting as needed.
    """
    interactDefaults.runButtonTemplate = template


def funcToParamDict(func, title=None, **overrides):
    """
    Converts a function into a list of child parameter dicts
    """
    children = []
    out = dict(name=func.__name__, type="group", children=children)
    if title is not None:
        out["title"] = _resolveTitle(func.__name__, title, forwardStrTitle=True)

    funcParams = inspect.signature(func).parameters
    if func.__doc__:
        # Reasonable "tip" default is the brief docstring description if it exists
        # Look for blank line that separates
        synopsis, _ = pydoc.splitdoc(func.__doc__)
        if synopsis:
            out.setdefault("tip", synopsis)

    # Make pyqtgraph parameter dicts from each parameter
    # Use list instead of funcParams.items() so kwargs can add to the iterable
    checkNames = list(funcParams)
    isKwarg = [p.kind is p.VAR_KEYWORD for p in funcParams.values()]
    if any(isKwarg):
        # Function accepts kwargs, so any overrides not already present as a function parameter should be accepted
        # Remove the keyword parameter since it can't be parsed properly
        # Only one kwarg can be in the signature, so there will be only one "True" index
        del checkNames[isKwarg.index(True)]
        notInSignature = [n for n in overrides if n not in checkNames]
        checkNames.extend(notInSignature)
    for name in checkNames:
        # May be none if this is an override name after function accepted kwargs
        param = funcParams.get(name)
        pgDict = createFuncParameter(name, param, overrides, title)
        children.append(pgDict)
    return out


def createFuncParameter(name, signatureParam, overridesDict, title=None):
    """
    Constructs a dict ready for insertion into a group parameter based on the provided information in the
    `inspect.signature` parameter, user-specified overrides, and true parameter name.

    Parameter signature information is considered the most "overridable", followed by documentation specifications.
    User overrides should be given the highest priority, i.e. not usurped by parameter default information.
    """
    if (
        signatureParam is not None
        and signatureParam.default is not signatureParam.empty
    ):
        # Maybe the user never specified type and value, since they can come directly from the default
        # Also, maybe override was a value without a type, so give a sensible default
        default = signatureParam.default
        signatureDict = {"value": default, "type": type(default).__name__}
    else:
        signatureDict = {}
    # Doc takes precedence over signature for any value information
    pgDict = signatureDict.copy()
    overrideInfo = overridesDict.get(name, {})
    if not isinstance(overrideInfo, dict):
        overrideInfo = {"value": overrideInfo}
    # Overrides take precedence over doc and signature
    pgDict.update(overrideInfo)
    # Name takes the highest precedence since it must be bindable to a function argument
    pgDict["name"] = name
    # Required function arguments with any override specifications can still be unfilled at this point
    pgDict.setdefault("value", RunOpts.PARAM_UNSET)

    # Anywhere a title is specified should take precedence over the default factory
    if title is not None:
        pgDict.setdefault("title", _resolveTitle(name, title))
    pgDict.setdefault("type", type(pgDict["value"]).__name__)
    return pgDict


class InteractiveFunction:
    """
    `interact` can be used with regular functions. However, when they are connected to changed or changing signals,
    there is no way to access these connections later to i.e. disconnect them temporarily. This utility class
    wraps a normal function but can provide an external scope for accessing the hooked up parameter signals.
    """

    def __init__(self, func, *, closures=None, **extra):
        """
        Wraps a callable function in a way that forwards Parameter arguments as keywords

        Parameters
        ----------
        func: callable
            Function to wrap
        closures: dict[str, callable]
            Arguments that shouldn't be constant, but can't be represented as a parameter. See the rst docs for
            more information.
        extra: dict
            extra keyword arguments to pass to `func` when this wrapper is called
        """
        super().__init__()
        self.params = []
        self.func = func
        if closures is None:
            closures = {}
        self.closures = closures
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        functools.update_wrapper(self, func)
        self._disconnected = False
        self.propagateParamChanges = False
        self.extra = extra
        self.paramKwargs = {}

    def __call__(self, **kwargs):
        """
        Calls `self.func`. Extra, closures, and parameter keywords as defined on init and through
        :func:`InteractiveFunction.setParams` are forwarded during the call.
        """
        if self.propagateParamChanges:
            self._updateParamsFromRunKwargs(**kwargs)

        runKwargs = self.extra.copy()
        runKwargs.update(self.paramKwargs)
        for kk, vv in self.closures.items():
            runKwargs[kk] = vv()
        runKwargs.update(**kwargs)
        return self.func(**runKwargs)

    def updateCachedParamValues(self, param, value):
        """
        This function is connected to `sigChanged` of every parameter associated with it. This way, those parameters
        don't have to be queried for their value every time InteractiveFunction is __call__'ed
        """
        self.paramKwargs[param.name()] = value

    def _updateParamsFromRunKwargs(self, **kwargs):
        """
        Updates attached params from __call__ without causing additional function runs
        """
        # Ensure updates don't cause firing of self's function
        wasDisconnected = self.disconnect()
        try:
            for param in self.params:
                name = param.name()
                if name in kwargs:
                    param.setValue(kwargs[name])
        finally:
            if not wasDisconnected:
                self.reconnect()

    def hookupParams(self, params=None, clearOld=True):
        """
        Binds a new set of parameters to this function. If `clearOld` is *True* (default), previously bound parameters
        are disconnected.

        Parameters
        ----------
        params: Sequence[Parameter]
            New parameters to listen for updates and optionally propagate keywords
            passed to :meth:`__call__`
        clearOld: bool
            If ``True``, previoulsy hooked up parameters will be removed first
        """
        if clearOld:
            self.removeParams()
        for param in params:
            # Weakref prevents elongating the life of parameters
            self.params.append(param)
            param.sigValueChanged.connect(self.updateCachedParamValues)
            # Populate initial values
            self.paramKwargs[param.name()] = param.value()

    def removeParams(self, clearCache=True):
        """
        Disconnects from all signals of parameters in `self.params`. Also, optionally clears the old cache of param
        values
        """
        for p in self.params:
            p.sigValueChanged.disconnect(self.updateCachedParamValues)
        # Disconnected all old signals, clear out and get ready for new ones
        self.params.clear()
        if clearCache:
            self.paramKwargs.clear()

    def runFromChangedOrChanging(self, param, value):
        if self._disconnected:
            return
        # Since this request came from a parameter, ensure it's not propagated back
        # for efficiency and to avoid ``changing`` signals causing ``changed`` values
        oldPropagate = self.propagateParamChanges
        self.propagateParamChanges = False
        try:
            ret = self(**{param.name(): value})
        finally:
            self.propagateParamChanges = oldPropagate
        return ret

    def runFromButton(self, **kwargs):
        if self._disconnected:
            return
        return self(**kwargs)

    def disconnect(self):
        """Simulates disconnecting the runnable by turning `runFrom*` functions into no-ops"""
        oldDisconnect = self._disconnected
        self._disconnected = True
        return oldDisconnect

    def reconnect(self):
        """Simulates reconnecting the runnable by re-enabling `runFrom*` functions"""
        oldDisconnect = self._disconnected
        self._disconnected = False
        return oldDisconnect

    def __str__(self):
        return self.func.__str__()


def _resolveTitle(name, titleFormat, forwardStrTitle=False):
    isstr = isinstance(titleFormat, str)
    if titleFormat is None or (isstr and not forwardStrTitle):
        return name
    elif isstr:
        return titleFormat
    # else: titleFormat should be callable
    return titleFormat(name)


def interact(
    func,
    *,
    ignores=None,
    runOpts=RunOpts.PARAM_UNSET,
    parent=RunOpts.PARAM_UNSET,
    title=RunOpts.PARAM_UNSET,
    nest=RunOpts.PARAM_UNSET,
    existOk=RunOpts.PARAM_UNSET,
    **overrides,
):
    """
    Interacts with a function by making Parameters for each argument.

    There are several potential use cases and argument handling possibilities depending on which values are
    passed to this function, so a more detailed explanation of several use cases is provided in
    the "Interactive Parameters" doc.

    if any non-defaults exist, a value must be
    provided for them in `descrs`. If this value should *not* be made into a parameter, include its name in `ignores`.

    Parameters
    ----------
    func: Callable
        function with which to interact. Can also be a :class:`InteractiveFunction`, if a reference to the bound
        signals is required.
    runOpts: `GroupParameter.<RUN_BUTTON, CHANGED, or CHANGING>` value
        How the function should be run, i.e. when pressing a button, on sigValueChanged, and/or on sigValueChanging
    ignores: Sequence
        Names of function arguments which shouldn't have parameters created
    parent: GroupParameter
        Parent in which to add arguemnt Parameters. If *None*, a new group parameter is created.
    title: str or Callable
        Title of the group sub-parameter if one must be created (see `nest` behavior). If a function is supplied, it
        must be of the form (str) -> str and will be passed the function name as an input
    nest: bool
        If *True*, the interacted function is given its own GroupParameter, and arguments to that function are
        'nested' inside as its children. If *False*, function arguments are directly added to this parameter
        instead of being placed inside a child GroupParameter
    existOk: bool
        Whether it is OK for existing paramter names to bind to this function. See behavior during
        'Parameter.insertChild'
    overrides:
        Override descriptions to provide additional parameter options for each argument. Moreover,
        extra parameters can be defined here if the original function uses ``**`` to consume additional keyword
        arguments. Each override can be a value (e.g. 5) or a dict specification of a parameter
        (e.g. dict(type='list', limits=[0, 10, 20]))
    """
    # Get every overridden default
    if runOpts is RunOpts.PARAM_UNSET:
        runOpts = interactDefaults.runOpts
    if parent is RunOpts.PARAM_UNSET:
        parent = interactDefaults.parent
    if title is RunOpts.PARAM_UNSET:
        title = interactDefaults.title
    if nest is RunOpts.PARAM_UNSET:
        nest = interactDefaults.nest
    if existOk is RunOpts.PARAM_UNSET:
        existOk = interactDefaults.existOk

    funcDict = funcToParamDict(func, title=title, **overrides)
    children = funcDict.pop("children", [])  # type: list[dict]
    chNames = [ch["name"] for ch in children]
    parent = _resolveParent(parent, nest, funcDict, existOk)

    if not isinstance(func, InteractiveFunction):
        # If a reference isn't captured somewhere, garbage collection of the newly created
        # "InteractiveFunction" instance prevents connected signals from firing
        # Use a list in case multiple interact() calls are made with the same function
        interactive = InteractiveFunction(func)
        if hasattr(func, "interactiveRefs"):
            func.interactiveRefs.append(interactive)
        else:
            func.interactiveRefs = [interactive]
        func = interactive

    # Values can't come both from closures and overrides/params, so ensure they don't get created
    if ignores is None:
        ignores = []
    ignores = list(ignores) + list(func.closures)

    # Recycle ignored content that is needed as a value
    recycleNames = set(overrides) & set(ignores) & set(chNames)
    extra = {key: overrides[key] for key in recycleNames}
    func.extra.update(extra)

    useParams = []
    for chDict in children:
        name = chDict["name"]
        if name in ignores:
            continue
        child = _createFuncParamChild(parent, chDict, runOpts, existOk, func)
        useParams.append(child)

    func.hookupParams(useParams)
    # If no top-level parent and no nesting, return the list of child parameters
    ret = parent or useParams
    if RunOpts.ON_BUTTON in runOpts:
        # Add an extra button child which can activate the function
        button = _makeRunButton(nest, funcDict.get("tip"), func)
        # Return just the button if no other params were allowed
        if not parent.hasChildren():
            ret = button
        parent.addChild(button, existOk=existOk)

    # Keep reference to avoid `func` getting garbage collected and allow later access
    return ret


def _resolveParent(parent, nest, parentOpts, existOk):
    """
    Returns parent parameter that holds function children. May be ``None`` if
    no top parent is provided and nesting is disabled.
    """
    funcGroup = parent
    if nest:
        funcGroup = Parameter.create(**parentOpts)
    if parent and nest:
        parent.addChild(funcGroup, existOk=existOk)
    return funcGroup


def _createFuncParamChild(parent, chDict, runOpts, existOk, toExec):
    name = chDict["name"]
    # Make sure args without defaults have overrides
    if (
        chDict["value"] is RunOpts.PARAM_UNSET
        and name not in toExec.closures
        and name not in toExec.extra
    ):
        raise ValueError(
            f"Cannot interact with `{toExec}` since it has required parameter `{name}`"
            f" with no default or closure value provided."
        )
    child = Parameter.create(**chDict)
    if parent:
        parent.addChild(child, existOk=existOk)
    if RunOpts.ON_CHANGED in runOpts:
        child.sigValueChanged.connect(toExec.runFromChangedOrChanging)
    if RunOpts.ON_CHANGING in runOpts:
        child.sigValueChanging.connect(toExec.runFromChangedOrChanging)
    return child


def _makeRunButton(nest, tip, interactiveFunc):
    # Add an extra button child which can activate the function
    template = interactDefaults.runButtonTemplate
    createOpts = template.copy()

    defaultName = template.get("defaultName", "Run")
    name = defaultName if nest else interactiveFunc.func.__name__
    createOpts.setdefault("name", name)
    if tip:
        createOpts["tip"] = tip
    child = Parameter.create(**createOpts)
    # A local function will avoid garbage collection by holding a reference to `toExec`
    child.sigActivated.connect(interactiveFunc.runFromButton)
    return child
