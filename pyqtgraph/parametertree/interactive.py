import contextlib
import functools
import inspect
import pydoc

from . import Parameter
from .parameterTypes import ActionGroupParameter
from .. import functions as fn


class PARAM_UNSET:
    """Sentinel value for detecting parameters with unset values"""


class RunOptions:

    ON_ACTION = "action"
    """
    Indicator for ``interactive`` parameter which adds an ``action`` parameter
    and runs when ``sigActivated`` is emitted.
    """
    ON_CHANGED = "changed"
    """
    Indicator for ``interactive`` parameter which runs the function every time one
    ``sigValueChanged`` is emitted from any of the parameters
    """
    ON_CHANGING = "changing"
    """
    Indicator for ``interactive`` parameter which runs the function every time one
     ``sigValueChanging`` is emitted from any of the parameters
    """


class InteractiveFunction:
    """
    ``interact`` can be used with regular functions. However, when they are connected to
    changed or changing signals, there is no way to access these connections later to
    i.e. disconnect them temporarily. This utility class wraps a normal function but
    can provide an external scope for accessing the hooked up parameter signals.
    """

    # Attributes below are populated by `update_wrapper` but aren't detected by linters
    __name__: str
    __qualname__: str

    def __init__(self, function, *, closures=None, **extra):
        """
        Wraps a callable function in a way that forwards Parameter arguments as keywords

        Parameters
        ----------
        function: callable
            Function to wrap
        closures: dict[str, callable]
            Arguments that shouldn't be constant, but can't be represented as a parameter.
            See the rst docs for more information.
        extra: dict
            extra keyword arguments to pass to ``function`` when this wrapper is called
        """
        super().__init__()
        self.parameters = {}
        self.extra = extra
        self.function = function
        if closures is None:
            closures = {}
        self.closures = closures
        self._disconnected = False
        self.parametersNeedRunKwargs = False
        self.parameterCache = {}

        # No need for wrapper __dict__ to function as function.__dict__, since
        # Only __doc__, __name__, etc. attributes are required
        functools.update_wrapper(self, function, updated=())

    def __call__(self, **kwargs):
        """
        Calls ``self.function``. Extra, closures, and parameter keywords as defined on
        init and through :func:`InteractiveFunction.setParams` are forwarded during the
        call.
        """
        if self.parametersNeedRunKwargs:
            self._updateParametersFromRunKwargs(**kwargs)

        runKwargs = self.extra.copy()
        runKwargs.update(self.parameterCache)
        for kk, vv in self.closures.items():
            runKwargs[kk] = vv()
        runKwargs.update(**kwargs)
        return self.function(**runKwargs)

    def updateCachedParameterValues(self, param, value):
        """
        This function is connected to ``sigChanged`` of every parameter associated with
        it. This way, those parameters don't have to be queried for their value every
        time InteractiveFunction is __call__'ed
        """
        self.parameterCache[param.name()] = value

    def _updateParametersFromRunKwargs(self, **kwargs):
        """
        Updates attached params from __call__ without causing additional function runs
        """
        # Ensure updates don't cause firing of self's function
        wasDisconnected = self.disconnect()
        try:
            for kwarg in set(kwargs).intersection(self.parameters):
                self.parameters[kwarg].setValue(kwargs[kwarg])
        finally:
            if not wasDisconnected:
                self.reconnect()

        for extraKey in set(kwargs) & set(self.extra):
            self.extra[extraKey] = kwargs[extraKey]

    def _disconnectParameter(self, param):
        param.sigValueChanged.disconnect(self.updateCachedParameterValues)
        for signal in (param.sigValueChanging, param.sigValueChanged):
            fn.disconnect(signal, self.runFromChangedOrChanging)

    def hookupParameters(self, params=None, clearOld=True):
        """
        Binds a new set of parameters to this function. If ``clearOld`` is *True* (
        default), previously bound parameters are disconnected.

        Parameters
        ----------
        params: Sequence[Parameter]
            New parameters to listen for updates and optionally propagate keywords
            passed to :meth:`__call__`
        clearOld: bool
            If ``True``, previously hooked up parameters will be removed first
        """
        if clearOld:
            self.removeParameters()
        for param in params:
            self.parameters[param.name()] = param
            param.sigValueChanged.connect(self.updateCachedParameterValues)
            # Populate initial values
            self.parameterCache[param.name()] = param.value()

    def removeParameters(self, clearCache=True):
        """
        Disconnects from all signals of parameters in ``self.parameters``. Also,
        optionally clears the old cache of param values
        """
        for p in self.parameters.values():
            self._disconnectParameter(p)
        # Disconnected all old signals, clear out and get ready for new ones
        self.parameters.clear()
        if clearCache:
            self.parameterCache.clear()

    def runFromChangedOrChanging(self, param, value):
        if self._disconnected:
            return None
        # Since this request came from a parameter, ensure it's not propagated back
        # for efficiency and to avoid ``changing`` signals causing ``changed`` values
        oldPropagate = self.parametersNeedRunKwargs
        self.parametersNeedRunKwargs = False
        try:
            ret = self(**{param.name(): value})
        finally:
            self.parametersNeedRunKwargs = oldPropagate
        return ret

    def runFromAction(self, **kwargs):
        if self._disconnected:
            return None
        return self(**kwargs)

    def disconnect(self):
        """
        Simulates disconnecting the runnable by turning ``runFrom*`` functions into no-ops
        """
        oldDisconnect = self._disconnected
        self._disconnected = True
        return oldDisconnect

    def setDisconnected(self, disconnected):
        """
        Sets the disconnected state of the runnable, see :meth:`disconnect` and
        :meth:`reconnect` for more information
        """
        oldDisconnect = self._disconnected
        self._disconnected = disconnected
        return oldDisconnect

    def reconnect(self):
        """Simulates reconnecting the runnable by re-enabling ``runFrom*`` functions"""
        oldDisconnect = self._disconnected
        self._disconnected = False
        return oldDisconnect

    def __str__(self):
        return f"{type(self).__name__}(`<{self.function.__name__}>`) at {hex(id(self))}"

    def __repr__(self):
        return (
            str(self) + " with keys:\n"
            f"parameters={list(self.parameters)}, "
            f"extra={list(self.extra)}, "
            f"closures={list(self.closures)}"
        )


class Interactor:
    runOptions = RunOptions.ON_ACTION
    parent = None
    titleFormat = None
    nest = True
    existOk = True
    runActionTemplate = dict(type="action", defaultName="Run")

    _optionNames = [
        "runOptions",
        "parent",
        "titleFormat",
        "nest",
        "existOk",
        "runActionTemplate",
    ]

    def __init__(self, **kwargs):
        """
        Initializes an Interactor with initial keyword arguments which can be anything
        accepted by :meth:`setOpts`
        """
        self.setOpts(**kwargs)

    def setOpts(self, **opts):
        """
        Overrides the default options for this interactor.

        Note! This method should only be used if you spawn your own Interactor; do not
        call it on ``defaultInteractor``. Instead, use ``defaultInteractor.optsContext``,
        which is guaranteed to revert to the default options when the context expires.

        Parameters
        ----------
        opts
            Keyword arguments to override the default options

        Returns
        -------
            dict of previous options that were overridden. This is useful for resetting
            the options afterward.
        """
        oldOpts = self.getOpts()
        allowed = set(oldOpts)
        errors = set(opts).difference(allowed)
        if errors:
            raise KeyError(f"Unrecognized options: {errors}. Must be one of: {allowed}")

        toReturn = {}
        toUse = {}
        for kk, vv in opts.items():
            toReturn[kk] = oldOpts[kk]
            toUse[kk] = vv
        self.__dict__.update(toUse)
        return toReturn

    @contextlib.contextmanager
    def optsContext(self, **opts):
        """
        Creates a new context for ``opts``, where each is reset to the old value
        when the context expires

        Parameters
        ----------
        opts:
            Options to set, must be one of the keys in :attr:`_optNames`
        """
        oldOpts = self.setOpts(**opts)
        yield
        self.setOpts(**oldOpts)

    def interact(
        self,
        function,
        *,
        ignores=None,
        runOptions=PARAM_UNSET,
        parent=PARAM_UNSET,
        titleFormat=PARAM_UNSET,
        nest=PARAM_UNSET,
        runActionTemplate=PARAM_UNSET,
        existOk=PARAM_UNSET,
        **overrides,
    ):
        """
        Interacts with a function by making Parameters for each argument.

        There are several potential use cases and argument handling possibilities
        depending on which values are passed to this function, so a more detailed
        explanation of several use cases is provided in the "Interactive Parameters" doc.

        if any non-defaults exist, a value must be provided for them in ``overrides``. If
        this value should *not* be made into a parameter, include its name in ``ignores``.

        Parameters
        ----------
        function: Callable
            function with which to interact. Can also be a :class:`InteractiveFunction`,
            if a reference to the bound signals is required.
        runOptions: ``GroupParameter.<RUN_ACTION, CHANGED, or CHANGING>`` value
            How the function should be run, i.e. when pressing an action, on
            sigValueChanged, and/or on sigValueChanging
        ignores: Sequence
            Names of function arguments which shouldn't have parameters created
        parent: GroupParameter
            Parent in which to add argument Parameters. If *None*, a new group
            parameter is created.
        titleFormat: str or Callable
            title of the group sub-parameter if one must be created (see ``nest``
            behavior). If a function is supplied, it must be of the form (str) -> str
            and will be passed the function name as an input
        nest: bool
            If *True*, the interacted function is given its own GroupParameter,
            and arguments to that function are 'nested' inside as its children.
            If *False*, function arguments are directly added to this parameter
            instead of being placed inside a child GroupParameter
        runActionTemplate: dict
            Template for the action parameter which runs the function, used
            if ``runOptions`` is set to ``GroupParameter.RUN_ACTION``. Note that
            if keys like "name" or "type" are not included, they are inferred
            from the previous / default ``runActionTemplate``. This allows
            items that should only be set per-function to exist here, like
            a ``shortcut`` or ``icon``.
        existOk: bool
            Whether it is OK for existing parameter names to bind to this function.
            See behavior during 'Parameter.insertChild'
        overrides:
            Override descriptions to provide additional parameter options for each
            argument. Moreover, extra parameters can be defined here if the original
            function uses ``**`` to consume additional keyword arguments. Each
            override can be a value (e.g. 5) or a dict specification of a
            parameter (e.g. dict(type='list', limits=[0, 10, 20]))
        """
        # Special case: runActionTemplate can be overridden to specify action
        if runActionTemplate is not PARAM_UNSET:
            runActionTemplate = {**self.runActionTemplate, **runActionTemplate}
        # Get every overridden default
        locs = locals()
        # Everything until action template
        opts = {kk: locs[kk] for kk in self._optionNames if locs[kk] is not PARAM_UNSET}
        oldOpts = self.setOpts(**opts)
        # Delete explicitly since correct values are now ``self`` attributes
        del runOptions, titleFormat, nest, existOk, parent, runActionTemplate

        function = self._toInteractiveFunction(function)
        funcDict = self.functionToParameterDict(function.function, **overrides)
        children = funcDict.pop("children", [])  # type: list[dict]
        chNames = [ch["name"] for ch in children]
        funcGroup = self._resolveFunctionGroup(funcDict, function)

        # Values can't come both from closures and overrides/params, so ensure they don't
        # get created
        if ignores is None:
            ignores = []
        ignores = list(ignores) + list(function.closures)

        # Recycle ignored content that is needed as a value
        recycleNames = set(ignores) & set(chNames)
        for name in recycleNames:
            value = children[chNames.index(name)]["value"]
            if name not in function.extra and value is not PARAM_UNSET:
                function.extra[name] = value

        missingChildren = [
            ch["name"]
            for ch in children
            if ch["value"] is PARAM_UNSET
            and ch["name"] not in function.closures
            and ch["name"] not in function.extra
        ]
        if missingChildren:
            # setOpts will not be called due to the value error, so reset here.
            # This only matters to restore Interactor state in an outer try-except
            # block
            self.setOpts(**oldOpts)
            raise ValueError(
                f"Cannot interact with `{function}` since it has required parameters "
                f"{missingChildren} with no default or closure values provided."
            )

        useParams = []
        checkNames = [n for n in chNames if n not in ignores]
        for name in checkNames:
            childOpts = children[chNames.index(name)]
            child = self.resolveAndHookupParameterChild(funcGroup, childOpts, function)
            if child is not None:
                useParams.append(child)

        function.hookupParameters(useParams)
        if RunOptions.ON_ACTION in self.runOptions:
            # Add an extra action child which can activate the function
            action = self._resolveRunAction(function, funcGroup, funcDict.get("tip"))
            if action:
                useParams.append(action)
        retValue = funcGroup if self.nest else useParams
        self.setOpts(**oldOpts)
        # Return either the parent which contains all added options, or the list
        # of created children (if no parent was created)
        return retValue

    @functools.wraps(interact)
    def __call__(self, function, **kwargs):
        return self.interact(function, **kwargs)

    def decorate(self, **kwargs):
        """
        Calls :meth:`interact` and returns the :class:`InteractiveFunction`.

        Parameters
        ----------
        kwargs
            Keyword arguments to pass to :meth:`interact`
        """

        def decorator(function):
            if not isinstance(function, InteractiveFunction):
                function = InteractiveFunction(function)
            self.interact(function, **kwargs)
            return function

        return decorator

    def _nameToTitle(self, name, forwardStringTitle=False):
        """
        Converts a function name to a title based on ``self.titleFormat``.

        Parameters
        ----------
        name: str
            Name of the function
        forwardStringTitle: bool
            If ``self.titleFormat`` is a string and ``forwardStrTitle`` is True,
            ``self.titleFormat`` will be used as the title. Otherwise, if
            ``self.titleFormat`` is *None*, the name will be returned unchanged.
            Finally, if ``self.titleFormat`` is a callable, it will be called with
            the name as an input and the output will be returned
        """
        titleFormat = self.titleFormat
        isString = isinstance(titleFormat, str)
        if titleFormat is None or (isString and not forwardStringTitle):
            return name
        elif isString:
            return titleFormat
        # else: titleFormat should be callable
        return titleFormat(name)

    def _resolveFunctionGroup(self, functionDict, interactiveFunction):
        """
        Returns parent parameter that holds function children. May be ``None`` if
        no top parent is provided and nesting is disabled.
        """
        funcGroup = self.parent
        if self.nest:
            funcGroup = Parameter.create(**functionDict)
            if self.parent:
                funcGroup = self.parent.addChild(funcGroup, existOk=self.existOk)
            funcGroup.sigActivated.connect(interactiveFunction.runFromAction)
        return funcGroup

    @staticmethod
    def _toInteractiveFunction(function):
        if isinstance(function, InteractiveFunction):
            # Nothing to do
            return function

        # If a reference isn't captured somewhere, garbage collection of the newly created
        # "InteractiveFunction" instance prevents connected signals from firing
        # Use a list in case multiple interact() calls are made with the same function
        interactive = InteractiveFunction(function)
        refOwner = function if not inspect.ismethod(function) else function.__func__
        if hasattr(refOwner, "interactiveRefs"):
            refOwner.interactiveRefs.append(interactive)
        else:
            refOwner.interactiveRefs = [interactive]
        return interactive

    def resolveAndHookupParameterChild(
        self, functionGroup, childOpts, interactiveFunction
    ):
        if not functionGroup:
            child = Parameter.create(**childOpts)
        else:
            child = functionGroup.addChild(childOpts, existOk=self.existOk)
        if RunOptions.ON_CHANGED in self.runOptions:
            child.sigValueChanged.connect(interactiveFunction.runFromChangedOrChanging)
        if RunOptions.ON_CHANGING in self.runOptions:
            child.sigValueChanging.connect(interactiveFunction.runFromChangedOrChanging)
        return child

    def _resolveRunAction(self, interactiveFunction, functionGroup, functionTip):
        if isinstance(functionGroup, ActionGroupParameter):
            functionGroup.setButtonOpts(visible=True)
            child = None
        else:
            # Add an extra action child which can activate the function
            createOpts = self._makePopulatedActionTemplate(
                interactiveFunction.__name__, functionTip
            )
            child = Parameter.create(**createOpts)
            child.sigActivated.connect(interactiveFunction.runFromAction)
            if functionGroup:
                functionGroup.addChild(child, existOk=self.existOk)
        return child

    def _makePopulatedActionTemplate(self, functionName="", functionTip=None):
        createOpts = self.runActionTemplate.copy()

        defaultName = createOpts.get("defaultName", "Run")
        name = defaultName if self.nest else functionName
        createOpts.setdefault("name", name)
        if functionTip:
            createOpts.setdefault("tip", functionTip)
        return createOpts

    def functionToParameterDict(self, function, **overrides):
        """
        Converts a function into a list of child parameter dicts
        """
        children = []
        name = function.__name__
        btnOpts = dict(**self._makePopulatedActionTemplate(name), visible=False)
        out = dict(name=name, type="_actiongroup", children=children, button=btnOpts)
        if self.titleFormat is not None:
            out["title"] = self._nameToTitle(name, forwardStringTitle=True)

        funcParams = inspect.signature(function).parameters
        if function.__doc__:
            # Reasonable "tip" default is the brief docstring description if it exists
            synopsis, _ = pydoc.splitdoc(function.__doc__)
            if synopsis:
                out.setdefault("tip", synopsis)
                out["button"].setdefault("tip", synopsis)

        # Make pyqtgraph parameter dicts from each parameter
        # Use list instead of funcParams.items() so kwargs can add to the iterable
        checkNames = list(funcParams)
        parameterKinds = [p.kind for p in funcParams.values()]
        _positional = inspect.Parameter.VAR_POSITIONAL
        _keyword = inspect.Parameter.VAR_KEYWORD
        if _keyword in parameterKinds:
            # Function accepts kwargs, so any overrides not already present as a function
            # parameter should be accepted
            # Remove the keyword parameter since it can't be parsed properly
            # Kwargs must appear at the end of the parameter list
            del checkNames[-1]
            notInSignature = [n for n in overrides if n not in checkNames]
            checkNames.extend(notInSignature)
        if _positional in parameterKinds:
            # *args is also difficult to handle for key-value paradigm
            # and will mess with the logic for detecting whether any parameter
            # is left unfilled
            del checkNames[parameterKinds.index(_positional)]

        for name in checkNames:
            # May be none if this is an override name after function accepted kwargs
            param = funcParams.get(name)
            pgDict = self.createFunctionParameter(name, param, overrides.get(name, {}))
            children.append(pgDict)
        return out

    def createFunctionParameter(self, name, signatureParameter, overridesInfo):
        """
        Constructs a dict ready for insertion into a group parameter based on the
        provided information in the ``inspect.signature`` parameter, user-specified
        overrides, and true parameter name. Parameter signature information is
        considered the most "overridable", followed by documentation specifications.
        User overrides should be given the highest priority, i.e. not usurped by
        parameter default information.

        Parameters
        ----------
        name : str
            Name of the parameter, comes from function signature
        signatureParameter : inspect.Parameter
            Information from the function signature, parsed by ``inspect``
        overridesInfo : dict
            User-specified overrides for this parameter. Can be a dict of options
            accepted by :class:`~pyqtgraph.parametertree.Parameter` or a value
        """
        if (
            signatureParameter is not None
            and signatureParameter.default is not signatureParameter.empty
        ):
            # Maybe the user never specified type and value, since they can come
            # directly from the default Also, maybe override was a value without a
            # type, so give a sensible default
            default = signatureParameter.default
            signatureDict = {"value": default, "type": type(default).__name__}
        else:
            signatureDict = {}
        # Doc takes precedence over signature for any value information
        pgDict = signatureDict.copy()
        if not isinstance(overridesInfo, dict):
            overridesInfo = {"value": overridesInfo}
        # Overrides take precedence over doc and signature
        pgDict.update(overridesInfo)
        # Name takes the highest precedence since it must be bindable to a function
        # argument
        pgDict["name"] = name
        # Required function arguments with any override specifications can still be
        # unfilled at this point
        pgDict.setdefault("value", PARAM_UNSET)

        # Anywhere a title is specified should take precedence over the default factory
        if self.titleFormat is not None:
            pgDict.setdefault("title", self._nameToTitle(name))
        pgDict.setdefault("type", type(pgDict["value"]).__name__)
        return pgDict

    def __str__(self):
        return f"Interactor with opts: {self.getOpts()}"

    def __repr__(self):
        return str(self)

    def getOpts(self):
        return {attr: getattr(self, attr) for attr in self._optionNames}


interact = Interactor()
