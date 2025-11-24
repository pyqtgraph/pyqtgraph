__all__ = ["CtrlNode", "PlottingCtrlNode"]

from typing import Iterable, Optional, Callable, Sequence, Any
import numpy as np

from ..Terminal import Terminal
from ...Qt import QtCore, QtWidgets


from ...WidgetGroup import WidgetGroup
from ...widgets.ColorButton import ColorButton
from ...widgets.SpinBox import SpinBox


from ..Node import Node



def generateUi(opts: Iterable[Sequence]) -> tuple[QtWidgets.QWidget, WidgetGroup, dict[str, QtWidgets.QWidget]]:
    """Convenience function for generating common UI types"""
    widget = QtWidgets.QWidget()
    l = QtWidgets.QFormLayout()
    l.setSpacing(0)
    widget.setLayout(l)
    ctrls = {}
    row = 0
    for opt in opts:
        if len(opt) == 2:
            k, t = opt
            o = {}
        elif len(opt) == 3:
            k, t, o = opt
        else:
            raise Exception("Widget specification must be (name, type) or (name, type, {opts})")

        # clean out these options so they don't get sent to SpinBox
        hidden = o.pop('hidden', False)
        tip = o.pop('tip', None)

        w: QtWidgets.QWidget
        if t == 'intSpin':
            w = QtWidgets.QSpinBox()
            if 'max' in o:
                w.setMaximum(o['max'])
            if 'min' in o:
                w.setMinimum(o['min'])
            if 'value' in o:
                w.setValue(o['value'])
        elif t == 'doubleSpin':
            w = QtWidgets.QDoubleSpinBox()
            if 'max' in o:
                w.setMaximum(o['max'])
            if 'min' in o:
                w.setMinimum(o['min'])
            if 'value' in o:
                w.setValue(o['value'])
        elif t == 'spin':
            w = SpinBox()
            w.setOpts(**o)
        elif t == 'check':
            w = QtWidgets.QCheckBox()
            if 'checked' in o:
                w.setChecked(o['checked'])
        elif t == 'combo':
            w = QtWidgets.QComboBox()
            for i in o['values']:
                w.addItem(i)
        # elif t == 'colormap':
        #     w = ColorMapper()
        elif t == 'color':
            w = ColorButton()
        else:
            raise Exception("Unknown widget type '%s'" % str(t))

        if tip is not None:
            w.setToolTip(tip)
        w.setObjectName(k)
        l.addRow(k, w)
        if hidden:
            w.hide()
            label = l.labelForField(w)
            label.hide()

        ctrls[k] = w
        w.rowNum = row
        row += 1
    group = WidgetGroup(widget)
    return widget, group, ctrls


class CtrlNode(Node):
    """Abstract class for nodes with auto-generated control UI"""

    sigStateChanged = QtCore.Signal(object)

    def __init__(self, name: str, ui: Optional[Sequence] = None, terminals: Optional[dict[str, dict]] = None) -> None:
        if terminals is None:
            terminals = {'In': {'io': 'in'}, 'Out': {'io': 'out', 'bypass': 'In'}}
        super().__init__(name=name, terminals=terminals)

        if ui is None:
            if hasattr(self, 'uiTemplate'):
                ui = self.uiTemplate
            else:
                ui = []

        self.ui, self.stateGroup, self.ctrls = generateUi(ui)
        self.stateGroup.sigChanged.connect(self.changed)

    def ctrlWidget(self) -> QtWidgets.QWidget:
        return self.ui

    def changed(self) -> None:
        self.update()
        self.sigStateChanged.emit(self)

    def process(self, In: dict, display: bool = True) -> dict:
        out = self.processData(In)
        return {'Out': out}

    def saveState(self) -> dict:
        state = Node.saveState(self)
        state['ctrl'] = self.stateGroup.state()
        return state

    def restoreState(self, state: dict) -> None:
        Node.restoreState(self, state)
        if self.stateGroup is not None:
            self.stateGroup.setState(state.get('ctrl', {}))

    def hideRow(self, name: str) -> None:
        w = self.ctrls[name]
        l = self.ui.layout().labelForField(w)
        w.hide()
        l.hide()

    def showRow(self, name: str) -> None:
        w = self.ctrls[name]
        l = self.ui.layout().labelForField(w)
        w.show()
        l.show()


class PlottingCtrlNode(CtrlNode):
    """Abstract class for CtrlNodes that can connect to plots."""

    def __init__(self, name: str, ui: Optional[Sequence] = None, terminals: Optional[dict[str, dict]] = None):
        # print "PlottingCtrlNode.__init__ called."
        super().__init__(name, ui=ui, terminals=terminals)
        self.plotTerminal = self.addOutput('plot', optional=True)

    def connected(self, term: Terminal, remote: Terminal) -> None:
        CtrlNode.connected(self, term, remote)
        if term is not self.plotTerminal:
            return
        node = remote.node()
        node.sigPlotChanged.connect(self.connectToPlot)
        self.connectToPlot(node)

    def disconnected(self, term: Terminal, remote: Terminal) -> None:
        CtrlNode.disconnected(self, term, remote)
        if term is not self.plotTerminal:
            return
        remote.node().sigPlotChanged.disconnect(self.connectToPlot)
        self.disconnectFromPlot(remote.node().getPlot())

    def connectToPlot(self, node: Optional[Node]) -> None:
        """Define what happens when the node is connected to a plot"""
        raise Exception("Must be re-implemented in subclass")

    def disconnectFromPlot(self, plot: Any) -> None:
        """Define what happens when the node is disconnected from a plot"""
        raise Exception("Must be re-implemented in subclass")

    def process(self, In: dict, display: bool = True) -> dict:
        out = CtrlNode.process(self, In, display)
        out['plot'] = None
        return out

