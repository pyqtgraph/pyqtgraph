import re
from contextlib import ExitStack

from ... import functions as fn
from ...Qt import QtCore, QtWidgets
from ...SignalProxy import SignalProxy
from ...widgets.BrushPreviewLabel import BrushPreviewLabel
from . import GroupParameterItem
from .basetypes import GroupParameter, Parameter, ParameterItem
from .qtenum import QtEnumParameter


class BrushParameterItem(GroupParameterItem):
    def __init__(self, param, depth):
        self.ctrlBtn = self.makeCtrlButton()
        super().__init__(param, depth)
        self.itemWidget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.brushLabel = BrushPreviewLabel(param)
        for child in self.brushLabel, self.ctrlBtn:
            layout.addWidget(child)
        self.itemWidget.setLayout(layout)

    def optsChanged(self, param, opts):
        if "enabled" in opts or "readonly" in opts:
            self.updateCtrlButton()

    def treeWidgetChanged(self):
        ParameterItem.treeWidgetChanged(self)
        tw = self.treeWidget()
        if tw is None:
            return
        tw.setItemWidget(self, 1, self.itemWidget)

    def valueChanged(self, param, val):
        self.updateCtrlButton()


def cap_first(s: str):
    if not s:
        return s
    return s[0].upper() + s[1:]


class BrushParameter(GroupParameter):
    """
    Controls the appearance of a QBrush value — the mirror image of
    PenParameter.

    When `saveState` is called, the value is encoded as (color, style).

    ============== ========================================================
    **Options:**
    color          brush color, can be any argument accepted by :func:`~pyqtgraph.mkColor` (defaults to black)
    style          String version of QBrushStyle enum, i.e. 'SolidPattern' (default), 'NoBrush', etc.
    ============== ========================================================
    """
    itemClass = BrushParameterItem

    def __init__(self, **opts):
        self.brush = fn.mkBrush(**opts)
        children = self._makeChildren(self.brush)
        if 'children' in opts:
            raise KeyError('Cannot set "children" argument in Brush Parameter opts')
        super().__init__(**opts, children=list(children))
        self.valChangingProxy = SignalProxy(
            self.sigValueChanging,
            delay=1.0,
            slot=self._childrenFinishedChanging,
            threadSafe=False,
        )

    @QtCore.Slot(object)
    def _childrenFinishedChanging(self, paramAndValue):
        self.setValue(self.brush)

    def setDefault(self, val, **kwargs):
        brush = self._interpretValue(val)
        with self.treeChangeBlocker():
            for opt in self.names:
                attrName = opt
                self.child(opt).setDefault(getattr(brush, attrName)(), **kwargs)
            out = super().setDefault(val, **kwargs)
        return out

    def saveState(self, filter=None):
        state = super().saveState(filter)
        opts = state.pop('children')
        state['value'] = tuple(o['value'] for o in opts.values())
        if 'default' not in state:
            state['default'] = state['value']  # TODO remove this after January 2025 (matches PenParameter)
        return state

    def restoreState(self, state, recursive=True, addChildren=True, removeChildren=True, blockSignals=True):
        return super().restoreState(state, recursive=False, addChildren=False, removeChildren=False, blockSignals=blockSignals)

    def _interpretValue(self, v):
        return self.mkBrush(v)

    def setValue(self, value, blockSignal=None):
        if not fn.eq(value, self.brush):
            value = self.mkBrush(value)
            self.updateFromBrush(self, value)
        return super().setValue(self.brush, blockSignal)

    def applyOptsToBrush(self, **opts):
        paramNames = set(opts).intersection(self.names)
        with self.treeChangeBlocker():
            if 'value' in opts:
                brush = self.mkBrush(opts.pop('value'))
                if not fn.eq(brush, self.brush):
                    self.updateFromBrush(self, brush)
            brushOpts = {}
            for kk in paramNames:
                brushOpts[kk] = opts[kk]
                self[kk] = opts[kk]
        return brushOpts

    def setOpts(self, **opts):
        if self.applyOptsToBrush(**opts):
            self.setValue(self.brush)
        return super().setOpts(**opts)

    def mkBrush(self, *args, **kwargs):
        """Thin wrapper around fn.mkBrush which accepts the serialized state from saveState"""
        if len(args) == 1 and isinstance(args[0], tuple) and len(args[0]) == len(self.childs):
            opts = dict(zip(self.names, args[0]))
            self.applyOptsToBrush(**opts)
            args = (self.brush,)
            kwargs = {}
        return fn.mkBrush(*args, **kwargs)

    def _makeChildren(self, boundBrush=None):
        bs = QtCore.Qt.BrushStyle
        param = Parameter.create(
            name='Params', type='group', children=[
                dict(name='color', type='color', value='k'),
                QtEnumParameter(bs, name='style', value='SolidPattern'),
            ]
        )

        optsBrush = boundBrush or fn.mkBrush('k')
        for p in param:
            name = p.name()
            attrName = name
            default = getattr(optsBrush, attrName)()
            replace = r'\1 \2'
            name = re.sub(r'(\w)([A-Z])', replace, name)
            name = name.title().strip()
            p.setOpts(title=name, default=default)

        if boundBrush is not None:
            self.updateFromBrush(param, boundBrush)
            for p in param:
                setName = f'set{cap_first(p.name())}'
                setattr(boundBrush, setName, p.setValue)
                newSetter = self.brushPropertySetter
                if p.type() != 'color':
                    p.sigValueChanging.connect(newSetter)
                p.sigValueChanged.disconnect(p._emitValueChanged)
                p.sigValueChanged.connect(newSetter)

        return param

    def brushPropertySetter(self, p, value):
        boundBrush = self.brush
        setName = f'set{cap_first(p.name())}'
        getattr(boundBrush.__class__, setName)(boundBrush, value)
        self.sigValueChanging.emit(self, boundBrush)

    @staticmethod
    def updateFromBrush(param, brush):
        """
        Applies settings from a brush to either a Parameter or dict. The Parameter or dict must already
        be populated with the relevant keys ('color', 'style').
        """
        stack = ExitStack()
        if isinstance(param, Parameter):
            names = param.names
            stack.enter_context(param.treeChangeBlocker())
        else:
            names = param
        for opt in names:
            attrName = opt
            param[opt] = getattr(brush, attrName)()
        stack.close()
