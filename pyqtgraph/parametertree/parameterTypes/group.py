from .. import ParameterItem, Parameter
from ... import functions as fn
from ...Qt import QtWidgets, QtCore, QtGui
from ... import icons

class GroupParameterItem(ParameterItem):
    """
    Group parameters are used mainly as a generic parent item that holds (and groups!) a set
    of child parameters. It also provides a simple mechanism for displaying a button or combo
    that can be used to add new parameters to the group.
    """
    def __init__(self, param, depth):
        ParameterItem.__init__(self, param, depth)
        self.updateDepth(depth)

        self.addItem = None
        if 'addText' in param.opts:
            addText = param.opts['addText']
            if 'addList' in param.opts:
                self.addWidget = QtWidgets.QComboBox()
                self.addWidget.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
                self.updateAddList()
                self.addWidget.currentIndexChanged.connect(self.addChanged)
            else:
                self.addWidget = QtWidgets.QPushButton(addText)
                self.addWidget.clicked.connect(self.addClicked)
            w = QtWidgets.QWidget()
            l = QtWidgets.QHBoxLayout()
            l.setContentsMargins(0,0,0,0)
            w.setLayout(l)
            l.addWidget(self.addWidget)
            l.addStretch()
            self.addWidgetBox = w
            self.addItem = QtWidgets.QTreeWidgetItem([])
            self.addItem.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            self.addItem.depth = self.depth + 1
            ParameterItem.addChild(self, self.addItem)
            self.addItem.setSizeHint(0, self.addWidgetBox.sizeHint())

        self.optsChanged(self.param, self.param.opts)

    def updateDepth(self, depth):
        ## Change item's appearance based on its depth in the tree
        ## This allows highest-level groups to be displayed more prominently.
        if depth == 0:
            for c in [0,1]:
                self.setBackground(c, QtGui.QBrush(QtGui.QColor(100,100,100)))
                self.setForeground(c, QtGui.QBrush(QtGui.QColor(220,220,255)))
                font = self.font(c)
                font.setBold(True)
                font.setPointSize(font.pointSize()+1)
                self.setFont(c, font)
        else:
            for c in [0,1]:
                self.setBackground(c, QtGui.QBrush(QtGui.QColor(220,220,220)))
                self.setForeground(c, QtGui.QBrush(QtGui.QColor(50,50,50)))
                font = self.font(c)
                font.setBold(True)
                #font.setPointSize(font.pointSize()+1)
                self.setFont(c, font)
        self.titleChanged()  # sets the size hint for column 0 which is based on the new font

    def addClicked(self):
        """Called when "add new" button is clicked
        The parameter MUST have an 'addNew' method defined.
        """
        self.param.addNew()

    def addChanged(self):
        """Called when "add new" combo is changed
        The parameter MUST have an 'addNew' method defined.
        """
        if self.addWidget.currentIndex() == 0:
            return
        typ = self.addWidget.currentText()
        self.param.addNew(typ)
        self.addWidget.setCurrentIndex(0)

    def treeWidgetChanged(self):
        ParameterItem.treeWidgetChanged(self)
        tw = self.treeWidget()
        if tw is None:
            return
        self.setFirstColumnSpanned(True)
        if self.addItem is not None:
            tw.setItemWidget(self.addItem, 0, self.addWidgetBox)
            self.addItem.setFirstColumnSpanned(True)

    def addChild(self, child):  ## make sure added childs are actually inserted before add btn
        if self.addItem is not None:
            ParameterItem.insertChild(self, self.childCount()-1, child)
        else:
            ParameterItem.addChild(self, child)

    def optsChanged(self, param, opts):
        ParameterItem.optsChanged(self, param, opts)

        if 'addList' in opts:
            self.updateAddList()

        if hasattr(self, 'addWidget'):
            if 'enabled' in opts:
                self.addWidget.setEnabled(opts['enabled'])

            if 'tip' in opts:
                self.addWidget.setToolTip(opts['tip'])

    def updateAddList(self):
        self.addWidget.blockSignals(True)
        try:
            self.addWidget.clear()
            self.addWidget.addItem(self.param.opts['addText'])
            for t in self.param.opts['addList']:
                self.addWidget.addItem(t)
        finally:
            self.addWidget.blockSignals(False)


class GroupParameter(Parameter):
    """
    Group parameters are used mainly as a generic parent item that holds (and groups!) a set
    of child parameters. 
    
    It also provides a simple mechanism for displaying a button or combo
    that can be used to add new parameters to the group. To enable this, the group 
    must be initialized with the 'addText' option (the text will be displayed on
    a button which, when clicked, will cause addNew() to be called). If the 'addList'
    option is specified as well, then a dropdown-list of addable items will be displayed
    instead of a button.
    """
    itemClass = GroupParameterItem

    sigAddNew = QtCore.Signal(object, object)  # self, type

    RUN_BUTTON = 'button'
    """Indicator for `interactive` parameter which runs the function on pressing a button parameter"""
    RUN_CHANGED = 'changed'
    """
    Indicator for `interactive` parameter which runs the function every time one `sigValueChanged` is emitted from
    any of the parameters
    """
    RUN_CHANGING = 'changing'
    """
    Indicator for `interactive` parameter which runs the function every time one `sigValueChanging` is emitted from
    any of the parameters
    """

    defaultRunOpts = RUN_CHANGED
    """Default behavior for running"""
    runTitleFormat = None
    """
    Formatter to create a parameter title from its name when using `Parameter.interact.` If not *None*, must be
    a callable of the form (name: str) -> str 
    """

    UNSET = object()
    """Sentinel value for detecting parameters with unset values"""

    def addNew(self, typ=None):
        """
        This method is called when the user has requested to add a new item to the group.
        By default, it emits ``sigAddNew(self, typ)``.
        """
        self.sigAddNew.emit(self, typ)

    def setAddList(self, vals):
        """Change the list of options available for the user to add to the group."""
        self.setOpts(addList=vals)

    def interactDecorator(self, **opts):
        """
        Decorator version of `Parameter.interact`. All options are forwarded there, except for `func` so it can be
        wrapped. Intended to be called using a GroupParameter, and the interactive parameter will be added
        as a child. Note that unless a parent is explicitly passed, it will be set to 'self'
        """

        def wrapper(func):
            opts.setdefault('parent', self)
            self.interact(func, **opts)
            return func

        return wrapper

    @classmethod
    def interact(cls, func, runOpts=None, ignores=None, deferred=None, parent=None, runFunc=None,
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
            function with which to interact
        runOpts: `GroupParameter.<RUN_BUTTON, CHANGED, or CHANGING>` value
            How the function should be run. If *None*, defaults to Parmeter.defaultRunOpts which can be set by the
            user.
        ignores: Sequence
            Names of function arguments which shouldn't have parameters created
        deferred: dict
            function arguments whose values should come from function evaluations rather than Parameters
            (must be a function that accepts no inputs and returns the desired value). This is helpful for providing
            external variables as function arguments, while making sure they are up to date.
        parent: GroupParameter
            Parent in which to add arguemnt Parameters. If *None*, a new group parameter is created.
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
        funcDict = cls._funcToParamDict(func, **overrides)
        children = funcDict.pop('children')

        if runOpts is None:
            runOpts = cls.defaultRunOpts
        if parent is None or nest:
            parentOpts = funcDict
            if cls.runTitleFormat is not None:
                parentOpts['title'] = cls.runTitleFormat(parentOpts['name'])
            host = Parameter.create(**parentOpts)
            if parent is not None:
                # Parent was provided and nesting is enabled, so place created args inside the nested GroupParmeter
                parent.addChild(host, existOk=existOk)
            parent = host

        if deferred is None:
            deferred = {}
        # Values can't come both from deferred and overrides/params, so ensure they don't get created
        if ignores is None:
            ignores = []
        ignores = list(ignores) + list(deferred)

        toExec = runFunc or func

        checkNames = [ch['name'] for ch in children if ch['name'] not in ignores]
        def runFunc(**extra):
            kwargs = {p.name(): p.value() for p in parent if p.name() in checkNames}
            for kk, vv in deferred.items():
                kwargs[kk] = vv()
            kwargs.update(**extra)
            return toExec(**kwargs)

        def runFunc_changing(_param, value):
            return runFunc(**{_param.name(): value})

        for chDict in children:
            name = chDict['name']
            # Make sure args without defaults have overrides
            if chDict['value'] is cls.UNSET and name not in deferred:
                raise ValueError(f'Cannot interact with "{func} since it has required parameter "{name}"'
                                 f' with no default or deferred value provided.')
            if name in ignores:
                # Don't make a parameter for this child
                # However, make sure to recycle their values if provided
                if name in overrides:
                    # Use default arg to avoid loop binding issue
                    deferred.setdefault(name, lambda _n=name: overrides[_n])
                continue

            child = parent.addChild(chDict, existOk=existOk)
            if cls.RUN_CHANGED in runOpts:
                child.sigValueChanged.connect(runFunc)
            if cls.RUN_CHANGING in runOpts:
                child.sigValueChanging.connect(runFunc_changing)
            # createdChildren.append(child)

        ret = parent
        # It doesn't make sense to register a parameter-less function without a button-run, since it will never
        # run and didn't create any children... Should this be an error/warning?
        # if not createdChildren and cls.RUN_BUTTON not in runOpts:
        #     warnings.warn(f'Interacting with function "{parent.title()}", but it is not runnable'
        #                   f' by button and possesses no parameters, so this is a no-op.', UserWarning)
        if cls.RUN_BUTTON in runOpts:
            # Add an extra button child which can activate the function
            name = 'Run' if nest else func.__name__
            createOpts = dict(name=name, type='action')
            tip = funcDict.get('tip')
            if tip:
                createOpts['tip'] = tip
            child = cls.create(**createOpts)
            # Return just the button if no other params were allowed
            if not parent.hasChildren():
                ret = child
            parent.addChild(child, existOk=existOk)
            child.sigActivated.connect(runFunc)
        return ret

    @classmethod
    def _funcToParamDict(cls, func, **overrides):
        """
        Converts a function into a list of child parameter dicts
        """
        children = []
        out = dict(name=func.__name__, type='group', children=children)

        funcParams = inspect.signature(func).parameters
        parsedDoc = cls.parseIniDocstring(func.__doc__)
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
            pgDict = cls._createFuncParameter(name, param, parsedDoc, overrides)
            children.append(pgDict)
        return out

    @classmethod
    def _createFuncParameter(cls, name, signatureParam, docDict, overridesDict):
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
        pgDict.setdefault('value', cls.UNSET)

        # Anywhere a title is specified should take precedence over the default factory
        if cls.runTitleFormat and 'title' not in pgDict:
            pgDict['title'] = cls.runTitleFormat(name)
        pgDict.setdefault('type', type(pgDict['value']).__name__)
        return pgDict

    @classmethod
    @contextlib.contextmanager
    def interactiveOptsContext(cls, **opts):
        """
        Sets default title format and default run format temporarily, within the scope of the context manager
        """
        attrs = {'runTitleFormat', 'defaultRunOpts'}
        oldOpts = {opt: getattr(cls, opt) for opt in attrs}
        useOpts = set(opts) & attrs
        for useOpt in useOpts:
            setattr(cls, useOpt, opts[useOpt])
        yield
        for useOpt in useOpts:
            setattr(cls, useOpt, oldOpts[useOpt])

    @classmethod
    def _parseIniDocstring_docstringParser(cls, doc):
        """
        Use docstring_parser for a smarter version of the ini parser. Doesn't require [<arg>.options] headers
        and can handle more dynamic parsing cases
        """
        # Revert to basic method if ini headers are already present
        if not doc or '.options]\n' in doc:
            return cls._parseIniDocstring_basic(doc)
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
            update = GroupParameter._parseIniDocstring_basic(miniDoc)
            update.pop('func-description', None)
            out.update(update)
        return out

    @classmethod
    def parseIniDocstring(cls, doc):
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
            return cls._parseIniDocstring_docstringParser(doc)
        except ImportError:
            return cls._parseIniDocstring_basic(doc)

    @staticmethod
    def _parseIniDocstring_basic(doc):
        # Adding "[DEFAULT] to the beginning of the doc will consume non-parameter descriptions
        out = {}
        doc = doc or '[DEFAULT]'
        # :param: style documentation violates ini standards
        lines = [l[1:] if l.startswith(':') else l for l in doc.splitlines()]
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
