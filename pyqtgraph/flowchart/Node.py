__all__ = ["Node", "NodeGraphicsItem"]

import sys
import types
from collections import OrderedDict
from typing import Optional, Any, Callable, Union, Unpack

from .. import functions as fn
from ..GraphicsScene.mouseEvents import MouseClickEvent, MouseDragEvent
from ..debug import printExc
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Qt import QtCore, QtGui, QtWidgets
from .Terminal import Terminal

exc_type = Union[tuple[type[BaseException], BaseException, types.TracebackType], tuple[None, None, None]]
translate = QtCore.QCoreApplication.translate


def strDict(d: dict) -> dict:
    return dict([(str(k), v) for k, v in d.items()])


class Node(QtCore.QObject):
    """
    Node represents the basic processing unit of a flowchart. 
    A Node subclass implements at least:
    
    1) A list of input / output terminals and their properties
    2) a process() function which takes the names of input terminals as keyword arguments and returns a dict with the names of output terminals as keys.

    A flowchart thus consists of multiple instances of Node subclasses, each of which is connected
    to other by wires between their terminals. A flowchart is, itself, also a special subclass of Node.
    This allows Nodes within the flowchart to connect to the input/output nodes of the flowchart itself.

    Optionally, a node class can implement the ctrlWidget() method, which must return a QWidget (usually containing other widgets) that will be displayed in the flowchart control panel. Some nodes implement fairly complex control widgets, but most nodes follow a simple form-like pattern: a list of parameter names and a single value (represented as spin box, check box, etc..) for each parameter. To make this easier, the CtrlNode subclass allows you to instead define a simple data structure that CtrlNode will use to automatically generate the control widget.     """

    sigOutputChanged = QtCore.Signal(object)  # self
    sigClosed = QtCore.Signal(object)
    sigRenamed = QtCore.Signal(object, object)
    sigTerminalRenamed = QtCore.Signal(object, object)  # term, oldName
    sigTerminalAdded = QtCore.Signal(object, object)  # self, term
    sigTerminalRemoved = QtCore.Signal(object, object)  # self, term

    def __init__(
            self,
            name: str,
            terminals: Optional[dict[str, dict]] = None,
            allowAddInput: bool = False,
            allowAddOutput: bool = False,
            allowRemove: bool = True
    ) -> None:
        """
        ==============  ============================================================
        **Arguments:**
        name            The name of this specific node instance. It can be any 
                        string, but must be unique within a flowchart. Usually,
                        we simply let the flowchart decide on a name when calling
                        Flowchart.addNode(...)
        terminals       Dict-of-dicts specifying the terminals present on this Node.
                        Terminal specifications look like::

                            'inputTerminalName': {'io': 'in'}
                            'outputTerminalName': {'io': 'out'} 
                            
                        There are a number of optional parameters for terminals:
                        multi, pos, renamable, removable, multiable, bypass. See
                        the Terminal class for more information.
        allowAddInput   bool; whether the user is allowed to add inputs by the
                        context menu.
        allowAddOutput  bool; whether the user is allowed to add outputs by the
                        context menu.
        allowRemove     bool; whether the user is allowed to remove this node by the
                        context menu.
        ==============  ============================================================  
        
        """
        QtCore.QObject.__init__(self)
        self._name: str = name
        self._bypass: bool = False
        self.bypassButton: QtWidgets.QPushButton | None = None  ## this will be set by the flowchart ctrl widget..
        self._graphicsItem: Optional[NodeGraphicsItem] = None
        self.terminals: dict[str, Terminal] = OrderedDict()
        self._inputs: dict = OrderedDict()
        self._outputs: dict = OrderedDict()
        self._allowAddInput: bool = allowAddInput  ## flags to allow the user to add/remove terminals
        self._allowAddOutput: bool = allowAddOutput
        self._allowRemove: bool = allowRemove

        self.exception: Optional[exc_type] = None
        if terminals is None:
            return
        for name, opts in terminals.items():
            self.addTerminal(name, **opts)

    def nextTerminalName(self, name: str) -> str:
        """Return an unused terminal name"""
        name2 = name
        i = 1
        while name2 in self.terminals:
            name2 = "%s.%d" % (name, i)
            i += 1
        return name2

    def addInput(self, name: str = "Input", **args: Unpack) -> Terminal:
        """Add a new input terminal to this Node with the given name. Extra
        keyword arguments are passed to Terminal.__init__.
        
        This is a convenience function that just calls addTerminal(io='in', ...)"""
        # print "Node.addInput called."
        return self.addTerminal(name, io='in', **args)

    def addOutput(self, name: str = "Output", **args: Unpack) -> Terminal:
        """Add a new output terminal to this Node with the given name. Extra
        keyword arguments are passed to Terminal.__init__.
        
        This is a convenience function that just calls addTerminal(io='out', ...)"""
        return self.addTerminal(name, io='out', **args)

    def removeTerminal(self, term: Union[Terminal, str]) -> None:
        """Remove the specified terminal from this Node. May specify either the 
        terminal's name or the terminal itself.
        
        Causes sigTerminalRemoved to be emitted."""
        if isinstance(term, Terminal):
            name = term.name()
        else:
            name = term
            term = self.terminals[name]

        # print "remove", name
        # term.disconnectAll()
        term.close()
        del self.terminals[name]
        if name in self._inputs:
            del self._inputs[name]
        if name in self._outputs:
            del self._outputs[name]
        self.graphicsItem().updateTerminals()
        self.sigTerminalRemoved.emit(self, term)

    def terminalRenamed(self, term: Terminal, oldName: str) -> None:
        """Called after a terminal has been renamed        
        
        Causes sigTerminalRenamed to be emitted."""
        newName = term.name()
        for d in [self.terminals, self._inputs, self._outputs]:
            if oldName not in d:
                continue
            d[newName] = d[oldName]
            del d[oldName]

        self.graphicsItem().updateTerminals()
        self.sigTerminalRenamed.emit(term, oldName)

    def addTerminal(self, name: str, **opts: Unpack) -> Terminal:
        """Add a new terminal to this Node with the given name. Extra
        keyword arguments are passed to Terminal.__init__.
                
        Causes sigTerminalAdded to be emitted."""
        name = self.nextTerminalName(name)
        term = Terminal(self, name, **opts)
        self.terminals[name] = term
        if term.isInput():
            self._inputs[name] = term
        elif term.isOutput():
            self._outputs[name] = term
        self.graphicsItem().updateTerminals()
        self.sigTerminalAdded.emit(self, term)
        return term

    def inputs(self) -> dict:
        """Return dict of all input terminals.
        Warning: do not modify."""
        return self._inputs

    def outputs(self) -> dict:
        """Return dict of all output terminals.
        Warning: do not modify."""
        return self._outputs

    def process(self, **kargs: Unpack) -> dict:
        """Process data through this node. This method is called any time the flowchart 
        wants the node to process data. It will be called with one keyword argument
        corresponding to each input terminal, and must return a dict mapping the name
        of each output terminal to its new value.
        
        This method is also called with a 'display' keyword argument, which indicates
        whether the node should update its display (if it implements any) while processing
        this data. This is primarily used to disable expensive display operations
        during batch processing.
        """
        return {}

    def graphicsItem(self) -> 'NodeGraphicsItem':
        """Return the GraphicsItem for this node. Subclasses may re-implement
        this method to customize their appearance in the flowchart."""
        if self._graphicsItem is None:
            self._graphicsItem = NodeGraphicsItem(self)
        return self._graphicsItem

    def __getitem__(self, item: str) -> Terminal:
        # return getattr(self, item)
        """Return the terminal with the given name"""
        if item not in self.terminals:
            raise KeyError(item)
        else:
            return self.terminals[item]

    def name(self) -> str:
        """Return the name of this node."""
        return self._name

    def rename(self, name: str) -> None:
        """Rename this node. This will cause sigRenamed to be emitted."""
        oldName = self._name
        self._name = name
        # self.emit(QtCore.SIGNAL('renamed'), self, oldName)
        self.sigRenamed.emit(self, oldName)

    def dependentNodes(self) -> set:
        """Return the list of nodes which provide direct input to this node"""
        nodes = set()
        for t in self.inputs().values():
            nodes |= set([i.node() for i in t.inputTerminals()])
        return nodes
        # return set([t.inputTerminals().node() for t in self.listInputs().values()])

    def __repr__(self) -> str:
        return "<Node %s @%x>" % (self.name(), id(self))

    def ctrlWidget(self) -> QtWidgets.QWidget | None:
        """Return this Node's control widget.
        
        By default, Nodes have no control widget. Subclasses may reimplement this 
        method to provide a custom widget. This method is called by Flowcharts
        when they are constructing their Node list."""
        return None

    def bypass(self, byp: bool) -> None:
        """Set whether this node should be bypassed.
        
        When bypassed, a Node's process() method is never called. In some cases,
        data is automatically copied directly from specific input nodes to 
        output nodes instead (see the bypass argument to Terminal.__init__). 
        This is usually called when the user disables a node from the flowchart 
        control panel.
        """
        self._bypass = byp
        if self.bypassButton is not None:
            self.bypassButton.setChecked(byp)
        self.update()

    def isBypassed(self) -> bool:
        """Return True if this Node is currently bypassed."""
        return self._bypass

    def setInput(self, **args: Unpack) -> None:
        """Set the values on input terminals. For most nodes, this will happen automatically through Terminal.inputChanged.
        This is normally only used for nodes with no connected inputs."""
        changed = False
        for k, v in args.items():
            term = self._inputs[k]
            oldVal = term.value()
            if not fn.eq(oldVal, v):
                changed = True
            term.setValue(v, process=False)
        if changed and '_updatesHandled_' not in args:
            self.update()

    def inputValues(self) -> dict:
        """Return a dict of all input values currently assigned to this node."""
        vals = {}
        for n, t in self.inputs().items():
            vals[n] = t.value()
        return vals

    def outputValues(self) -> dict:
        """Return a dict of all output values currently generated by this node."""
        vals = {}
        for n, t in self.outputs().items():
            vals[n] = t.value()
        return vals

    def connected(self, localTerm: Terminal, remoteTerm: Terminal) -> None:
        """Called whenever one of this node's terminals is connected elsewhere."""
        pass

    def disconnected(self, localTerm: Terminal, remoteTerm: Terminal) -> None:
        """Called whenever one of this node's terminals is disconnected from another."""
        pass

    def update(self, signal: bool = True) -> None:
        """Collect all input values, attempt to process new output values, and propagate downstream.
        Subclasses should call update() whenever thir internal state has changed
        (such as when the user interacts with the Node's control widget). Update
        is automatically called when the inputs to the node are changed.
        """
        vals = self.inputValues()
        # print "  inputs:", vals
        try:
            if self.isBypassed():
                out = self.processBypassed(vals)
            else:
                out = self.process(**strDict(vals))
            # print "  output:", out
            if out is not None:
                if signal:
                    self.setOutput(**out)
                else:
                    self.setOutputNoSignal(**out)
            for n, t in self.inputs().items():
                t.setValueAcceptable(True)
            self.clearException()
        except:
            # printExc( "Exception while processing %s:" % self.name())
            for n, t in self.outputs().items():
                t.setValue(None)
            self.setException(sys.exc_info())

            if signal:
                # self.emit(QtCore.SIGNAL('outputChanged'), self)  ## triggers flowchart to propagate new data
                self.sigOutputChanged.emit(self)  ## triggers flowchart to propagate new data

    def processBypassed(self, args: dict) -> dict:
        """Called when the flowchart would normally call Node.process, but this node is currently bypassed.
        The default implementation looks for output terminals with a bypass connection and returns the
        corresponding values. Most Node subclasses will _not_ need to reimplement this method."""
        result: dict = {}
        for term in list(self.outputs().values()):
            byp = term.bypassValue()
            if byp is None:
                result[term.name()] = None
            else:
                result[term.name()] = args.get(byp, None)
        return result

    def setOutput(self, **vals: Unpack) -> None:
        self.setOutputNoSignal(**vals)
        # self.emit(QtCore.SIGNAL('outputChanged'), self)  ## triggers flowchart to propagate new data
        self.sigOutputChanged.emit(self)  ## triggers flowchart to propagate new data

    def setOutputNoSignal(self, **vals: Unpack) -> None:
        for k, v in vals.items():
            term = self.outputs()[k]
            term.setValue(v)
            # targets = term.connections()
            # for t in targets:  ## propagate downstream
            # if t is term:
            # continue
            # t.inputChanged(term)
            term.setValueAcceptable(True)

    def setException(self, exc: Optional[exc_type]) -> None:
        self.exception = exc
        self.recolor()

    def clearException(self) -> None:
        self.setException(None)

    def recolor(self) -> None:
        if self.exception is None:
            self.graphicsItem().setPen(QtGui.QPen(QtGui.QColor(0, 0, 0)))
        else:
            self.graphicsItem().setPen(QtGui.QPen(QtGui.QColor(150, 0, 0), 3))

    def saveState(self) -> dict:
        """Return a dictionary representing the current state of this node
        (excluding input / output values). This is used for saving/reloading
        flowcharts. The default implementation returns this Node's position,
        bypass state, and information about each of its terminals. 
        
        Subclasses may want to extend this method, adding extra keys to the returned
        dict."""
        pos = self.graphicsItem().pos()
        state = {'pos': (pos.x(), pos.y()), 'bypass': self.isBypassed()}
        termsEditable = self._allowAddInput | self._allowAddOutput
        for term in list(self._inputs.values()) + list(self._outputs.values()):
            termsEditable |= term._renamable | term._removable | term._multiable
        if termsEditable:
            state['terminals'] = self.saveTerminals()
        return state

    def restoreState(self, state: dict) -> None:
        """Restore the state of this node from a structure previously generated
        by saveState(). """
        pos = state.get('pos', (0, 0))
        self.graphicsItem().setPos(*pos)
        self.bypass(state.get('bypass', False))
        if 'terminals' in state:
            self.restoreTerminals(state['terminals'])

    def saveTerminals(self) -> OrderedDict:
        terms = OrderedDict()
        for n, t in self.terminals.items():
            terms[n] = (t.saveState())
        return terms

    def restoreTerminals(self, state: dict) -> None:
        for name in list(self.terminals.keys()):
            if name not in state:
                self.removeTerminal(name)
        for name, opts in state.items():
            if name in self.terminals:
                term = self[name]
                term.setOpts(**opts)
                continue
            try:
                opts = strDict(opts)
                self.addTerminal(name, **opts)
            except:
                printExc("Error restoring terminal %s (%s):" % (str(name), str(opts)))

    def clearTerminals(self) -> None:
        for t in self.terminals.values():
            t.close()
        self.terminals = OrderedDict()
        self._inputs = OrderedDict()
        self._outputs = OrderedDict()

    def close(self) -> None:
        """Cleans up after the node--removes terminals, graphicsItem, widget"""
        self.disconnectAll()
        self.clearTerminals()
        item = self.graphicsItem()
        if item.scene() is not None:
            item.scene().removeItem(item)
        self._graphicsItem = None
        w = self.ctrlWidget()
        if w is not None:
            w.setParent(None)
        # self.emit(QtCore.SIGNAL('closed'), self)
        self.sigClosed.emit(self)

    def disconnectAll(self) -> None:
        for t in self.terminals.values():
            t.disconnectAll()


class TextItem(QtWidgets.QGraphicsTextItem):
    def __init__(self, text: str, parent: QtWidgets.QGraphicsItem, on_update: Optional[Callable]) -> None:
        super().__init__(text, parent)
        self.on_update = on_update

    def focusOutEvent(self, ev: QtGui.QFocusEvent | None) -> None:
        super().focusOutEvent(ev)
        if self.on_update is not None:
            self.on_update()

    def keyPressEvent(self, ev: QtGui.QKeyEvent | None) -> None:
        if ev is None:
            return super().keyPressEvent(ev)
        if ev.key() == QtCore.Qt.Key.Key_Enter or ev.key() == QtCore.Qt.Key.Key_Return:
            if self.on_update is not None:
                self.on_update()
                return
        super().keyPressEvent(ev)

    def mousePressEvent(self, ev: QtWidgets.QGraphicsSceneMouseEvent | None) -> None:
        if ev is None:
            return super().mousePressEvent(ev)
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextEditorInteraction)
            self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)  # focus text label
        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            self.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.NoTextInteraction)


# class NodeGraphicsItem(QtWidgets.QGraphicsItem):
class NodeGraphicsItem(GraphicsObject):
    def __init__(self, node: Node) -> None:
        # QtWidgets.QGraphicsItem.__init__(self)
        GraphicsObject.__init__(self)
        # QObjectWorkaround.__init__(self)

        # self.shadow = QtWidgets.QGraphicsDropShadowEffect()
        # self.shadow.setOffset(5,5)
        # self.shadow.setBlurRadius(10)
        # self.setGraphicsEffect(self.shadow)

        self.pen = fn.mkPen(0, 0, 0)
        self.selectPen = fn.mkPen(200, 200, 200, width=2)
        self.brush = fn.mkBrush(200, 200, 200, 150)
        self.hoverBrush = fn.mkBrush(200, 200, 200, 200)
        self.selectBrush = fn.mkBrush(200, 200, 255, 200)
        self.hovered = False

        self.node = node
        flags = self.GraphicsItemFlag.ItemIsMovable | self.GraphicsItemFlag.ItemIsSelectable | self.GraphicsItemFlag.ItemIsFocusable | self.GraphicsItemFlag.ItemSendsGeometryChanges
        # flags =  self.ItemIsFocusable |self.ItemSendsGeometryChanges

        self.setFlags(flags)
        self.bounds = QtCore.QRectF(0, 0, 100, 100)
        self.nameItem = TextItem(self.node.name(), self, self.labelChanged)
        self.nameItem.setDefaultTextColor(QtGui.QColor(50, 50, 50))
        self.nameItem.moveBy(self.bounds.width() / 2. - self.nameItem.boundingRect().width() / 2., 0)
        self._titleOffset = 25
        self._nodeOffset = 12
        self.updateTerminals()
        # self.setZValue(10)

        self.menu = QtWidgets.QMenu()
        self.buildMenu()

    def setTitleOffset(self, new_offset: int) -> None:
        """
        This method sets the rendering offset introduced after the title of the node.
        This method automatically updates the terminal labels. The default for this value is 25px.

        :param new_offset: The new offset to use in pixels at 100% scale.
        """
        self._titleOffset = new_offset
        self.updateTerminals()

    def titleOffset(self) -> int:
        """
        This method returns the current title offset in use.

        :returns: The offset in px.
        """
        return self._titleOffset

    def setTerminalOffset(self, new_offset: int) -> None:
        """
        This method sets the rendering offset introduced after every terminal of the node.
        This method automatically updates the terminal labels. The default for this value is 12px.

        :param new_offset: The new offset to use in pixels at 100% scale.
        """
        self._nodeOffset = new_offset
        self.updateTerminals()

    def terminalOffset(self) -> int:
        """
        This method returns the current terminal offset in use.

        :returns: The offset in px.
        """
        return self._nodeOffset

        # self.node.sigTerminalRenamed.connect(self.updateActionMenu)

    # def setZValue(self, z):
    # for t, item in self.terminals.values():
    # item.setZValue(z+1)
    # GraphicsObject.setZValue(self, z)

    def labelChanged(self) -> None:
        newName = self.nameItem.toPlainText()
        if newName != self.node.name():
            self.node.rename(newName)

        ### re-center the label
        bounds = self.boundingRect()
        self.nameItem.setPos(bounds.width() / 2. - self.nameItem.boundingRect().width() / 2., 0)

    def setPen(self, *args: Unpack, **kwargs: Unpack) -> None:
        self.pen = fn.mkPen(*args, **kwargs)
        self.update()

    def setBrush(self, brush: Callable) -> None:
        self.brush = brush
        self.update()

    def updateTerminals(self) -> None:
        self.terminals = {}
        inp = self.node.inputs()
        out = self.node.outputs()

        maxNode = max(len(inp), len(out))

        # calculate new height
        newHeight = self._titleOffset + maxNode * self._nodeOffset

        # if current height is not equal to new height, update
        if not self.bounds.height() == newHeight:
            self.bounds.setHeight(newHeight)
            self.update()

        # Populate inputs
        y = self._titleOffset
        for i, t in inp.items():
            item = t.graphicsItem()
            item.setParentItem(self)
            # item.setZValue(self.zValue()+1)
            item.setAnchor(0, y)
            self.terminals[i] = (t, item)
            y += self._nodeOffset

        # Populate inputs
        y = self._titleOffset
        for i, t in out.items():
            item = t.graphicsItem()
            item.setParentItem(self)
            item.setZValue(self.zValue())
            item.setAnchor(self.bounds.width(), y)
            self.terminals[i] = (t, item)
            y += self._nodeOffset

        # self.buildMenu()

    def boundingRect(self) -> QtCore.QRectF:
        return self.bounds.adjusted(-5, -5, 5, 5)

    def paint(self, p, *args) -> None:

        p.setPen(self.pen)
        if self.isSelected():
            p.setPen(self.selectPen)
            p.setBrush(self.selectBrush)
        else:
            p.setPen(self.pen)
            if self.hovered:
                p.setBrush(self.hoverBrush)
            else:
                p.setBrush(self.brush)

        p.drawRect(self.bounds)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent | None) -> None:
        if event is None:
            return None
        event.ignore()

    def mouseClickEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent | None) -> None:
        if event is None:
            return None
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            event.accept()
            sel = self.isSelected()
            self.setSelected(True)
            if not sel and self.isSelected():
                self.update()

        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            event.accept()
            self.raiseContextMenu(event)

    def mouseDragEvent(self, ev: QtWidgets.QGraphicsSceneMouseEvent | None) -> None:
        if ev is None:
            return
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            self.setPos(self.pos() + self.mapToParent(ev.pos()) - self.mapToParent(ev.lastPos()))

    def hoverEvent(self, ev) -> None:
        if not ev.isExit() and ev.acceptClicks(QtCore.Qt.MouseButton.LeftButton):
            ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton)
            self.hovered = True
        else:
            self.hovered = False
        self.update()

    def keyPressEvent(self, ev: QtGui.QKeyEvent | None) -> None:
        if ev is None:
            return None
        if ev.key() == QtCore.Qt.Key.Key_Delete or ev.key() == QtCore.Qt.Key.Key_Backspace:
            ev.accept()
            if not self.node._allowRemove:
                return
            self.node.close()
        else:
            ev.ignore()

    def itemChange(self, change: QtWidgets.QGraphicsItem.GraphicsItemChange, val: Any) -> None:
        if change == self.GraphicsItemChange.ItemPositionHasChanged:
            for k, t in self.terminals.items():
                t[1].nodeMoved()
        return GraphicsObject.itemChange(self, change, val)

    def getMenu(self) -> QtWidgets.QMenu:
        return self.menu

    def raiseContextMenu(self, ev: MouseClickEvent | MouseDragEvent | QtWidgets.QGraphicsSceneMouseEvent) -> None:
        menu = self.scene().addParentContextMenus(self, self.getMenu(), ev)
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(int(pos.x()), int(pos.y())))

    def buildMenu(self) -> None:
        self.menu.clear()
        self.menu.setTitle(translate("Context Menu", "Node"))
        a = self.menu.addAction(translate("Context Menu", "Add input"), self.addInputFromMenu)
        if not self.node._allowAddInput:
            a.setEnabled(False)
        a = self.menu.addAction(translate("Context Menu", "Add output"), self.addOutputFromMenu)
        if not self.node._allowAddOutput:
            a.setEnabled(False)
        a = self.menu.addAction(translate("Context Menu", "Remove node"), self.node.close)
        if not self.node._allowRemove:
            a.setEnabled(False)

    def addInputFromMenu(self) -> None:  ## called when add input is clicked in context menu
        self.node.addInput(renamable=True, removable=True, multiable=True)

    def addOutputFromMenu(self) -> None:  ## called when add output is clicked in context menu
        self.node.addOutput(renamable=True, removable=True, multiable=False)
