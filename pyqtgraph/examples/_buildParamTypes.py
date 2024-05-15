from _paramtreecfg import cfg

from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.Parameter import PARAM_TYPES
from pyqtgraph.parametertree.parameterTypes import GroupParameter

_encounteredTypes = {'group'}


def makeChild(chType, cfgDict):
    _encounteredTypes.add(chType)
    param = Parameter.create(name='widget', type=chType)
    if param.hasValue():
        param.setDefault(param.value())

    def setOpt(_param, _val):
        # Treat blank strings as "None" to allow 'unsetting' that option
        if isinstance(_val, str) and _val == '':
            _val = None
        param.setOpts(**{_param.name(): _val})

    optsChildren = []
    metaChildren = []
    for optName, optVals in cfgDict.items():
        child = Parameter.create(name=optName, **optVals)
        if ' ' in optName:
            metaChildren.append(child)
        else:
            optsChildren.append(child)
            child.sigValueChanged.connect(setOpt)
    # Populate initial options
    for p in optsChildren:
        setOpt(p, p.value() if p.hasValue() else None)

    grp = Parameter.create(
        name=f'Sample {chType.title()}', type='group', children=metaChildren + [param] + optsChildren
    )
    grp.setOpts(expanded=False)
    return grp


def makeMetaChild(name, cfgDict):
    children = []
    for chName, chOpts in cfgDict.items():
        if not isinstance(chOpts, dict):
            ch = Parameter.create(name=chName, type=chName, value=chOpts)
        else:
            ch = Parameter.create(name=chName, **chOpts)
        _encounteredTypes.add(ch.type())
        children.append(ch)
    param = Parameter.create(name=name, type='group', children=children)
    param.setOpts(expanded=False)
    return param


def makeAllParamTypes():
    children = []
    for name, paramCfg in cfg.items():
        if ' ' in name:
            children.append(makeMetaChild(name, paramCfg))
        else:
            children.append(makeChild(name, paramCfg))

    params = Parameter.create(name='Example Parameters', type='group', children=children)

    # Slider needs minor tweak
    sliderGrp = params.child('Sample Slider')
    slider = sliderGrp.child('widget')
    slider.setOpts(limits=[0, 100])

    # Also minor tweak to meta opts
    def setOpt(_param, _val):
        infoChild.setOpts(**{_param.name(): _val})
    meta = params.child('Applies to All Types')
    infoChild = meta.child('Extra Information')
    for child in meta.children()[1:]:
        child.sigValueChanged.connect(setOpt)

    def onChange(_param, _val):
        if _val == 'Use span':
            span = slider.opts.pop('span', None)
            slider.setOpts(span=span)
        else:
            limits = slider.opts.pop('limits', None)
            slider.setOpts(limits=limits)
    sliderGrp.child('How to Set').sigValueChanged.connect(onChange)

    def activate(action):
        for ch in params:
            if isinstance(ch, GroupParameter):
                ch.setOpts(expanded=action.name() == 'Expand All')

    for name in 'Collapse', 'Expand':
        btn = Parameter.create(name=f'{name} All', type='action')
        btn.sigActivated.connect(activate)
        params.insertChild(0, btn)
    missing = [
        typ
        for typ in set(PARAM_TYPES).difference(_encounteredTypes)
        if not typ.startswith("_")
    ]
    if missing:
        raise RuntimeError(f'{missing} parameters are not represented')
    return params
