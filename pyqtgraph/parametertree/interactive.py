import ast
import configparser
import contextlib
import functools
import inspect
import textwrap
import weakref
from functools import wraps

from . import Parameter

class RunOpts:
    class PARAM_UNSET: pass
    """Sentinel value for detecting parameters with unset values"""

    ON_BUTTON = 'button'
    """Indicator for `interactive` parameter which runs the function on pressing a button parameter"""
    ON_CHANGED = 'changed'
    """
    Indicator for `interactive` parameter which runs the function every time one `sigValueChanged` is emitted from
    any of the parameters
    """
    ON_CHANGING = 'changing'
    """
    Indicator for `interactive` parameter which runs the function every time one `sigValueChanging` is emitted from
    any of the parameters
    """

    defaultCfg = {}
    """Populated with any values that should override the default signature values in :func:`interact`."""

    @classmethod
    @contextlib.contextmanager
    def optsContext(cls, **opts):
        """
        Sets default title format and default run format temporarily, within the scope of the context manager
        """
        oldOpts = RunOpts.defaultCfg.copy()
        cls.setOpts(**opts)
        yield
        RunOpts.defaultCfg = oldOpts

    @classmethod
    def setOpts(cls, **opts):
        """Overrides the default behavior of function interaction"""
        attrs = set(inspect.signature(interact).parameters)
        optsSet = set(opts)
        useOpts = optsSet & attrs
        if any(optsSet.difference(attrs)):
            raise KeyError(f'Unrecognized option(s) "{optsSet.difference(attrs)}"')
        for useOpt in useOpts:
            if opts[useOpt] is None:
                cls.defaultCfg.pop(useOpt, None)
            else:
                cls.defaultCfg[useOpt] = opts[useOpt]


def funcToParamDict(func, title=None, **overrides):
    """
    Converts a function into a list of child parameter dicts
    """
    children = []
    out = dict(name=func.__name__, type='group', children=children)
    if title is not None:
        out['title'] = _resolveTitle(func.__name__, title)

    funcParams = inspect.signature(func).parameters
    parsedDoc = parseIniDocstring(func.__doc__)
    out.setdefault('tip', parsedDoc.get('func-description'))

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
        pgDict = createFuncParameter(name, param, parsedDoc, overrides, title)
        children.append(pgDict)
    return out


def createFuncParameter(name, signatureParam, docDict, overridesDict, title=None):
    """
    (Once organization PR is in place, this will be a regular function in the file instead of a class method).
    Constructs a dict ready for insertion into a group parameter based on the provided information in the
    `inspect.signature` parameter, user-specified overrides, function doc info, and true parameter name.

    Parameter signature information is considered the most "overridable", followed by documentation specifications.
    User overrides should be given the highest priority, i.e. not usurped by function doc values or parameter
    default information.
    """
    if signatureParam is not None and signatureParam.default is not signatureParam.empty:
        # Maybe the user never specified type and value, since they can come directly from the default
        # Also, maybe override was a value without a type, so give a sensible default
        default = signatureParam.default
        signatureDict = {'value': default,
                         'type': type(default).__name__}
    else:
        signatureDict = {}
    # Doc takes precedence over signature for any value information
    pgDict = {**signatureDict, **docDict.get(name, {})}
    overrideInfo = overridesDict.get(name, {})
    if not isinstance(overrideInfo, dict):
        overrideInfo = {'value': overrideInfo}
    # Overrides take precedence over doc and signature
    pgDict.update(overrideInfo)
    # Name takes highest precedence since it must be bindable to a function argument
    pgDict['name'] = name
    # Required function arguments with any override specifications can still be unfilled at this point
    pgDict.setdefault('value', RunOpts.PARAM_UNSET)

    # Anywhere a title is specified should take precedence over the default factory
    if title is not None:
        pgDict.setdefault('title', _resolveTitle(name, title))
    pgDict.setdefault('type', type(pgDict['value']).__name__)
    # Handle helpText, pType from PrjParam style
    if 'pType' in pgDict:
        pgDict['type'] = pgDict['pType']
    if 'helpText' in pgDict:
        pgDict['tip'] = pgDict['helpText']
    return pgDict


def parseIniDocstring(doc):
    """
    Parses function documentation for relevant parameter definitions.

    `doc` must be formatted like an .ini file, where each option's parameters are preceded by a [<arg>.options]
    header. See the examples in tests/parametertree/test_docparser for valid configurations. Note that if the
    `docstring_parser` module is available in the python environment, section headers as described above are
    *not required* since they can be inferred from the properly formatted docstring.

    The return value is a dict where each entry contains the found specifications of a given argument, i.e
    {"param1": {"limits": [0, 10], "title": "hi"}, "param2": {...}}, etc.

    Note that currently only literal evaluation with builtin objects is supported, i.e. the return result of
    ast.literal_eval on the value string of each option.

    Parameters
    ----------
    doc: str
      Documentation to parse
    """
    try:
        import docstring_parser
        return _parseIniDocstring_docstringParser(doc)
    except ImportError:
        return _parseIniDocstring_basic(doc)


def _parseIniDocstring_docstringParser(doc):
    """
    Use docstring_parser for a smarter version of the ini parser. Doesn't require [<arg>.options] headers
    and can handle more dynamic parsing cases
    """
    # Revert to basic method if ini headers are already present
    if not doc or '.options]\n' in doc:
        return _parseIniDocstring_basic(doc)
    import docstring_parser
    out = {}
    parsed = docstring_parser.parse(doc)
    out['func-description'] = '\n'.join([desc for desc in [parsed.short_description, parsed.long_description]
                                         if desc is not None])
    for param in parsed.params:
        # Construct mini ini file around each parameter
        header = f'[{param.arg_name}.options]'
        miniDoc = param.description
        if header not in miniDoc:
            miniDoc = f'{header}\n{miniDoc}'
        # top-level parameter no longer represents whole function
        update = _parseIniDocstring_basic(miniDoc)
        update.pop('func-description', None)
        out.update(update)
    return out


def _parseIniDocstring_basic(doc):
    # Adding "[DEFAULT] to the beginning of the doc will consume non-parameter descriptions
    out = {}
    doc = doc or '[DEFAULT]'
    # Account for several things in commonly supported docstring formats:
    # Indentation nesting in numpy style
    # :param: in rst
    lines = []
    for line in doc.splitlines():
        if line.startswith(':'):
            # :param: style documentation violates ini standards
            line = line[1:]
        lines.append(line)
    doc = textwrap.dedent('\n'.join(lines))
    if not doc.startswith('[DEFAULT]'):
        doc = '[DEFAULT]\n' + doc
    parser = configparser.ConfigParser(allow_no_value=True)
    # Save case sensitivity
    parser.optionxform = str
    try:
        parser.read_string(doc)
    except configparser.Error:
        # Many things can go wrong reading a badly-formatted docstring, so failsafe by returning early
        return out
    for kk, vv in parser.items():
        if not kk.endswith('.options'):
            continue
        paramName = kk.split('.')[0]
        # vv is a section with options for the parameter, but each option must be literal eval'd for non-string
        # values
        paramValues = dict(vv)
        # Consolidate all non-valued key strings into a single tip since they likely belong to the argument
        # documentation. Since dict preserves order, it should come back out in the right order.
        backupTip = None
        for paramK, paramV in list(paramValues.items()):
            if paramV is None:
                # Considered a tip of the current option
                if backupTip is None:
                    backupTip = ''
                backupTip = f'{backupTip} {paramK}'
                # Remove this from the return value since it isn't really meaninful
                del paramValues[paramK]
                continue
            try:
                paramValues[paramK] = ast.literal_eval(paramV)
            except:
                # There are many reasons this can fail, a safe fallback is the original string value
                pass
        if backupTip is not None:
            paramValues.setdefault('tip', backupTip.strip())
        out[paramName] = paramValues
    # Since function documentation can be used as a description for whatever group parameter hosts these
    # parameters, store it in a name guaranteed not to collide with parameter names since it's invalid
    # variable syntax (contains '-')
    out['func-description'] = '\n'.join(parser.defaults())
    return out


class InteractiveFunction:
    """
    `interact` can be used with regular functions. However, when they are connected to changed or changing signals,
    there is no way to access these connections later to i.e. disconnect them temporarily. This utility class
    wraps a normal function but can provide an external scope for accessing the hooked up parameter signals.
    """

    def __init__(self, func, deferred=None, **extra):
        """
        For information on these parameters, see the signature of :func:`interact`. `extra` are extra kwargs that aren't
        parameters, but are forwarded to `func`
        """
        super().__init__()
        self.params = []
        self.func = func
        if deferred is None:
            deferred = {}
        self.deferred = deferred
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        functools.update_wrapper(self, func)
        self._disconnected = False
        self.extra = extra
        self.paramKwargs = {}

    def __call__(self, **kwargs):
        """
        Calls `self.func`. Extra, deferred, and parameter keywords as defined on init and through
        :func:`InteractiveFunction.setParams` are forwarded during the call.
        """
        runKwargs = self.extra.copy()
        runKwargs.update(self.paramKwargs)
        for kk, vv in self.deferred.items():
            runKwargs[kk] = vv()
        runKwargs.update(**kwargs)
        return self.func(**runKwargs)

    def updateCachedParamValues(self, param, value):
        """
        This function is connected to `sigChanged` of every parameter associated with it. This way, those parameters
        don't have to be queried for their value every time InteractiveFunction is __call__'ed
        """
        self.paramKwargs[param.name()] = value

    def setParams(self, params=None):
        """Creates weakrefs to each parameter to avoid extending their lives"""
        if params is None:
            params = []
        for p in self.params:
            p = p()
            if p is None:
                raise RuntimeError('Calling interactive function with deleted parameter')
            p.sigValueChanged.disconnect(self.updateCachedParamValues)
        # Disconnected all old signals, clear out and get ready for new ones
        self.params.clear()
        # Also flush old cache
        self.paramKwargs.clear()
        for param in params:
            self.params.append(weakref.ref(param))
            param.sigValueChanged.connect(self.updateCachedParamValues)
            # Populate initial values
            self.paramKwargs[param.name()] = param.value()

    def runFromChangedOrChanging(self, _param, value):
        if self._disconnected:
            return
        return self(**{_param.name(): value})

    def runFromButton(self, **kwargs):
        if self._disconnected:
            return
        return self(**kwargs)

    def disconnect(self):
        """Simulates disconnecting the runnable by turning `runFrom*` functions into no-ops"""
        self._disconnected = True

    def reconnect(self):
        """Simulates reconnecting the runnable by re-enabling `runFrom*` functions"""
        self._disconnected = False

    def __str__(self):
        return self.func.__str__()

def _resolveTitle(name, titleFormat):
    if titleFormat is None:
        return name
    if isinstance(titleFormat, str):
        return titleFormat
    # else: titleFormat should be callable
    return titleFormat(name)

def _supplyCfgOpts(func):
    """Makes sure that config options can be supplied during normal function runs"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        newKwargs = {**RunOpts.defaultCfg, **kwargs}
        return func(*args, **newKwargs)
    return wrapper

@_supplyCfgOpts
def interact(func, *, runOpts=RunOpts.ON_CHANGED, ignores=None, parent=None, title=None, runFunc=None,
             nest=True, existOk=True, **overrides):
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
    runFunc: Callable
        Simplifies the process of interacting with a wrapped function without requiring `functools`. See the
        linked documentation for an example.
    nest: bool
        If *True*, the interacted function is given its own GroupParameter, and arguments to that function are
        'nested' inside as its children. If *False*, function arguments are directly added to this paremeter
        instead of being placed inside a child GroupParameter
    existOk: bool
        Whether it is OK for existing paramter names to bind to this function. See behavior during
        'Parameter.insertChild'
    overrides: sequence
        Override descriptions to provide additional parameter options for each argument. Moreover,
        extra parameters can be defined here if the original function uses ``**`` to consume additional keyword
        arguments. Each override can be a value (e.g. 5) or a dict specification of a parameter
        (e.g. dict(type='list', limits=[0, 10, 20]))
    """
    funcDict = funcToParamDict(func, title=title, **overrides)
    children = funcDict.pop('children', []) # type: list[dict]
    chNames = [ch['name'] for ch in children]
    parent = _resolveParent(parent, nest, funcDict, existOk)

    toExec = runFunc or func
    if not isinstance(toExec, InteractiveFunction):
        toExec = InteractiveFunction(toExec)

    # Values can't come both from deferred and overrides/params, so ensure they don't get created
    if ignores is None:
        ignores = []
    ignores = list(ignores) + list(toExec.deferred)

    # Recycle ignored content that is needed as a value
    recycleNames = set(overrides) & set(ignores) & set(chNames)
    extra = {key: overrides[key] for key in recycleNames}
    toExec.extra.update(extra)

    useParams = []
    for chDict in children:
        name = chDict['name']
        if name in ignores:
            continue
        child = _createFuncParamChild(parent, chDict, runOpts, existOk, toExec)
        useParams.append(child)

    toExec.setParams(useParams)
    ret = parent
    if RunOpts.ON_BUTTON in runOpts:
        # Add an extra button child which can activate the function
        button = _makeRunButton(nest, funcDict.get('tip'), toExec)
        # Return just the button if no other params were allowed
        if not parent.hasChildren():
            ret = button
        parent.addChild(button, existOk=existOk)

    # Keep reference to avoid `toExec` getting garbage collected and allow later access
    return ret

def _resolveParent(parent, nest, parentOpts, existOk):
    if parent is None or nest:
        host = Parameter.create(**parentOpts)
        if parent is not None:
            # Parent was provided and nesting is enabled, so place created args inside the nested GroupParmeter
            host = parent.addChild(host, existOk=existOk)
        parent = host
    return parent

def _createFuncParamChild(parent, chDict, runOpts, existOk, toExec):
    name = chDict['name']
    # Make sure args without defaults have overrides
    if chDict['value'] is RunOpts.PARAM_UNSET and name not in toExec.deferred and name not in toExec.extra:
        raise ValueError(f'Cannot interact with "{toExec} since it has required parameter "{name}"'
                         f' with no default or deferred value provided.')
    child = parent.addChild(chDict, existOk=existOk)
    # I tried connecting directly to the runnables in `toExec`, but they result in early garbage collection. This
    # doesn't happen with local functions
    def runner(param, val):
        toExec.runFromChangedOrChanging(param, val)
    if RunOpts.ON_CHANGED in runOpts:
        child.sigValueChanged.connect(runner)
    if RunOpts.ON_CHANGING in runOpts:
        child.sigValueChanging.connect(runner)
    return child

def _makeRunButton(nest, tip, interactiveFunc):
    # Add an extra button child which can activate the function
    name = 'Run' if nest else interactiveFunc.func.__name__
    createOpts = dict(name=name, type='action')
    if tip:
        createOpts['tip'] = tip
    child = Parameter.create(**createOpts)
    # A local function will avoid garbage collection by holding a reference to `toExec`
    def run():
        interactiveFunc.runFromButton()
    child.sigActivated.connect(run)
    return child
