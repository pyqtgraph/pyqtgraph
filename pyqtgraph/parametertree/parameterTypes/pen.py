import re
from contextlib import ExitStack

from . import GroupParameterItem, WidgetParameterItem
from .basetypes import GroupParameter, Parameter, ParameterItem
from .qtenum import QtEnumParameter
from ... import functions as fn
from ...Qt import QtCore, QtWidgets
from ...SignalProxy import SignalProxy
from ...widgets.PenPreviewLabel import PenPreviewLabel

class PenParameterItem(GroupParameterItem):
    def __init__(self, param, depth):
        self.defaultBtn = self.makeDefaultButton()
        super().__init__(param, depth)
        self.itemWidget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.penLabel = PenPreviewLabel(param)
        for child in self.penLabel, self.defaultBtn:
            layout.addWidget(child)
        self.itemWidget.setLayout(layout)

    def optsChanged(self, param, opts):
        if "enabled" in opts or "readonly" in opts:
            self.updateDefaultBtn()

    def treeWidgetChanged(self):
        ParameterItem.treeWidgetChanged(self)
        tw = self.treeWidget()
        if tw is None:
            return
        tw.setItemWidget(self, 1, self.itemWidget)

    defaultClicked = WidgetParameterItem.defaultClicked
    makeDefaultButton = WidgetParameterItem.makeDefaultButton

    def valueChanged(self, param, val):
        self.updateDefaultBtn()

    def updateDefaultBtn(self):
        self.defaultBtn.setEnabled(
            not self.param.valueIsDefault()
            and self.param.opts["enabled"]
            and self.param.writable()
        )


class PenParameter(GroupParameter):
    """
    Controls the appearance of a QPen value.

    When `saveState` is called, the value is encoded as (color, width, style, capStyle, joinStyle, cosmetic)

    ============== ========================================================
    **Options:**
    color          pen color, can be any argument accepted by :func:`~pyqtgraph.mkColor` (defaults to black)
    width          integer width >= 0 (defaults to 1)
    style          String version of QPenStyle enum, i.e. 'SolidLine' (default), 'DashLine', etc.
    capStyle       String version of QPenCapStyle enum, i.e. 'SquareCap' (default), 'RoundCap', etc.
    joinStyle      String version of QPenJoinStyle enum, i.e. 'BevelJoin' (default), 'RoundJoin', etc.
    cosmetic       Boolean, whether or not the pen is cosmetic (defaults to True)
    ============== ========================================================
    """
    itemClass = PenParameterItem

    def __init__(self, **opts):
        self.pen = fn.mkPen(**opts)
        children = self._makeChildren(self.pen)
        if 'children' in opts:
            raise KeyError('Cannot set "children" argument in Pen Parameter opts')
        super().__init__(**opts, children=list(children))
        self.valChangingProxy = SignalProxy(
            self.sigValueChanging,
            delay=1.0,
            slot=self._childrenFinishedChanging,
            threadSafe=False,
        )

    def _childrenFinishedChanging(self, paramAndValue):
        self.setValue(self.pen)

    def setDefault(self, val):
        pen = self._interpretValue(val)
        with self.treeChangeBlocker():
            # Block changes until all are finalized
            for opt in self.names:
                # Booleans have different naming convention
                if isinstance(self[opt], bool):
                    attrName = f'is{opt.title()}'
                else:
                    attrName = opt
                self.child(opt).setDefault(getattr(pen, attrName)())
            out = super().setDefault(val)
        return out

    def saveState(self, filter=None):
        state = super().saveState(filter)
        opts = state.pop('children')
        state['value'] = tuple(o['value'] for o in opts.values())
        return state

    def restoreState(self, state, recursive=True, addChildren=True, removeChildren=True, blockSignals=True):
        return super().restoreState(state, recursive=False, addChildren=False, removeChildren=False, blockSignals=blockSignals)

    def _interpretValue(self, v):
        return self.mkPen(v)

    def setValue(self, value, blockSignal=None):
        if not fn.eq(value, self.pen):
            value = self.mkPen(value)
            self.updateFromPen(self, value)
        return super().setValue(self.pen, blockSignal)

    def applyOptsToPen(self, **opts):
        # Transform opts into a value for the current pen
        paramNames = set(opts).intersection(self.names)
        # Value should be overridden by opts
        with self.treeChangeBlocker():
            if 'value' in opts:
                pen = self.mkPen(opts.pop('value'))
                if not fn.eq(pen, self.pen):
                    self.updateFromPen(self, pen)
            penOpts = {}
            for kk in paramNames:
                penOpts[kk] = opts[kk]
                self[kk] = opts[kk]
        return penOpts

    def setOpts(self, **opts):
        # Transform opts into a value
        penOpts = self.applyOptsToPen(**opts)
        if penOpts:
            self.setValue(self.pen)
        return super().setOpts(**opts)

    def mkPen(self, *args, **kwargs):
        """Thin wrapper around fn.mkPen which accepts the serialized state from saveState"""
        if len(args) == 1 and isinstance(args[0], tuple) and len(args[0]) == len(self.childs):
            opts = dict(zip(self.names, args[0]))
            self.applyOptsToPen(**opts)
            args = (self.pen,)
            kwargs = {}
        return fn.mkPen(*args, **kwargs)

    def _makeChildren(self, boundPen=None):
        cs = QtCore.Qt.PenCapStyle
        js = QtCore.Qt.PenJoinStyle
        ps = QtCore.Qt.PenStyle
        param = Parameter.create(
            name='Params', type='group', children=[
                dict(name='color', type='color', value='k'),
                dict(name='width', value=1, type='int', limits=[0, None]),
                QtEnumParameter(ps, name='style', value='SolidLine'),
                QtEnumParameter(cs, name='capStyle'),
                QtEnumParameter(js, name='joinStyle'),
                dict(name='cosmetic', type='bool', value=True)
            ]
            )

        optsPen = boundPen or fn.mkPen()
        for p in param:
            name = p.name()
            # Qt naming scheme uses isXXX for booleans
            if isinstance(p.value(), bool):
                attrName = f'is{name.title()}'
            else:
                attrName = name
            default = getattr(optsPen, attrName)()
            replace = r'\1 \2'
            name = re.sub(r'(\w)([A-Z])', replace, name)
            name = name.title().strip()
            p.setOpts(title=name, default=default)

        def penPropertyWrapper(propertySetter):
            def tiePenPropToParam(_, value):
                propertySetter(value)
                self.sigValueChanging.emit(self, self.pen)

            return tiePenPropToParam

        if boundPen is not None:
            self.updateFromPen(param, boundPen)
            for p in param:
                setter, setName = self._setterForParam(p.name(), boundPen, returnName=True)
                # Instead, set the parameter which will signal the old setter
                setattr(boundPen, setName, p.setValue)
                newSetter = penPropertyWrapper(setter)
                # Edge case: color picker uses a dialog with user interaction, so wait until full change there
                if p.type() != 'color':
                    p.sigValueChanging.connect(newSetter)
                # Force children to emulate self's value instead of being part of a tree like normal
                p.sigValueChanged.disconnect(p._emitValueChanged)
                # Some widgets (e.g. checkbox, combobox) don't emit 'changing' signals, so tie to 'changed' as well
                p.sigValueChanged.connect(newSetter)

        return param

    @staticmethod
    def _setterForParam(paramName, obj, returnName=False):
        formatted = paramName[0].upper() + paramName[1:]
        setter = getattr(obj, f'set{formatted}')
        if returnName:
            return setter, formatted
        return setter

    @staticmethod
    def updateFromPen(param, pen):
        """
        Applies settings from a pen to either a Parameter or dict. The Parameter or dict must already
        be populated with the relevant keys that can be found in `PenSelectorDialog.mkParam`.
        """
        stack = ExitStack()
        if isinstance(param, Parameter):
            names = param.names
            # Block changes until all are finalized
            stack.enter_context(param.treeChangeBlocker())
        else:
            names = param
        for opt in names:
            # Booleans have different naming convention
            if isinstance(param[opt], bool):
                attrName = f'is{opt.title()}'
            else:
                attrName = opt
            param[opt] = getattr(pen, attrName)()
        stack.close()
