# -*- coding: utf-8 -*-
from ..Qt import QtCore, QtGui
from ..python2_3 import asUnicode
from .Parameter import Parameter, registerParameterType
from .ParameterItem import ParameterItem
from ..widgets.SpinBox import SpinBox
from ..widgets.ColorButton import ColorButton
from ..colormap import ColorMap
from .. import icons as icons
from .. import functions as fn
from collections import OrderedDict
import os

from ..widgets.PenSelectorDialog import PenSelectorDialog

import numpy as np

class WidgetParameterItem(ParameterItem):
    """
    ParameterTree item with:
    
      * label in second column for displaying value
      * simple widget for editing value (displayed instead of label when item is selected)
      * button that resets value to default
    
    ==========================  =============================================================
    **Registered Types:**
    int                         Displays a :class:`SpinBox <pyqtgraph.SpinBox>` in integer
                                mode.
    float                       Displays a :class:`SpinBox <pyqtgraph.SpinBox>`.
    bool                        Displays a QCheckBox
    str                         Displays a QLineEdit
    color                       Displays a :class:`ColorButton <pyqtgraph.ColorButton>`
    colormap                    Displays a :class:`GradientWidget <pyqtgraph.GradientWidget>`
    ==========================  =============================================================
    
    This class can be subclassed by overriding makeWidget() to provide a custom widget.
    """
    def __init__(self, param, depth):
        ParameterItem.__init__(self, param, depth)

        self.asSubItem = False  # place in a child item's column 0 instead of column 1
        self.hideWidget = True  ## hide edit widget, replace with label when not selected
                                ## set this to False to keep the editor widget always visible
        
        # build widget with a display label and default button
        w = self.makeWidget()  
        self.widget = w
        self.eventProxy = EventProxy(w, self.widgetEventFilter)

        if self.asSubItem:
            self.subItem = QtGui.QTreeWidgetItem()
            self.subItem.depth = self.depth + 1
            self.subItem.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
            self.addChild(self.subItem)

        self.defaultBtn = QtGui.QPushButton()
        self.defaultBtn.setAutoDefault(False)
        self.defaultBtn.setFixedWidth(20)
        self.defaultBtn.setFixedHeight(20)
        modDir = os.path.dirname(__file__)
        self.defaultBtn.setIcon(icons.getGraphIcon('default'))
        self.defaultBtn.clicked.connect(self.defaultClicked)
        
        self.displayLabel = QtGui.QLabel()
        
        layout = QtGui.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        if not self.asSubItem:
            layout.addWidget(w, 1)
        layout.addWidget(self.displayLabel, 1)
        layout.addStretch(0)
        layout.addWidget(self.defaultBtn)
        self.layoutWidget = QtGui.QWidget()
        self.layoutWidget.setLayout(layout)
        
        if w.sigChanged is not None:
            w.sigChanged.connect(self.widgetValueChanged)
            
        if hasattr(w, 'sigChanging'):
            w.sigChanging.connect(self.widgetValueChanging)
            
        ## update value shown in widget. 
        opts = self.param.opts
        if opts.get('value', None) is not None:
            self.valueChanged(self, opts['value'], force=True)
        else:
            ## no starting value was given; use whatever the widget has
            self.widgetValueChanged()

        self.updateDefaultBtn()
        
        self.optsChanged(self.param, self.param.opts)

        # set size hints
        sw = self.widget.sizeHint()
        sb = self.defaultBtn.sizeHint()
        # shrink row heights a bit for more compact look
        sw.setHeight(int(sw.height() * 0.9))
        sb.setHeight(int(sb.height() * 0.9))
        if self.asSubItem:
            self.setSizeHint(1, sb)
            self.subItem.setSizeHint(0, sw)
        else:
            w = sw.width() + sb.width()
            h = max(sw.height(), sb.height())
            self.setSizeHint(1, QtCore.QSize(w, h))

    def makeWidget(self):
        """
        Return a single widget whose position in the tree is determined by the
        value of self.asSubItem. If True, it will be placed in the second tree
        column, and if False, the first tree column of a child item.

        The widget must be given three attributes:
        
        ==========  ============================================================
        sigChanged  a signal that is emitted when the widget's value is changed
        value       a function that returns the value
        setValue    a function that sets the value
        ==========  ============================================================
            
        This is a good function to override in subclasses.
        """
        opts = self.param.opts
        t = opts['type']
        if t in ('int', 'float'):
            defs = {
                'value': 0, 'min': None, 'max': None,
                'step': 1.0, 'dec': False, 
                'siPrefix': False, 'suffix': '', 'decimals': 3,
            }
            if t == 'int':
                defs['int'] = True
                defs['minStep'] = 1.0
            for k in defs:
                if k in opts:
                    defs[k] = opts[k]
            if 'limits' in opts:
                defs['min'], defs['max'] = opts['limits']
            w = SpinBox()
            w.setOpts(**defs)
            w.sigChanged = w.sigValueChanged
            w.sigChanging = w.sigValueChanging
        elif t == 'bool':
            w = QtGui.QCheckBox()
            w.sigChanged = w.toggled
            w.value = w.isChecked
            w.setValue = w.setChecked
            self.hideWidget = False
        elif t == 'str':
            w = QtGui.QLineEdit()
            w.setStyleSheet('border: 0px')
            w.sigChanged = w.editingFinished
            w.value = lambda: asUnicode(w.text())
            w.setValue = lambda v: w.setText(asUnicode(v))
            w.sigChanging = w.textChanged
        elif t == 'color':
            w = ColorButton()
            w.sigChanged = w.sigColorChanged
            w.sigChanging = w.sigColorChanging
            w.value = w.color
            w.setValue = w.setColor
            self.hideWidget = False
            w.setFlat(True)
        elif t == 'colormap':
            from ..widgets.GradientWidget import GradientWidget ## need this here to avoid import loop
            w = GradientWidget(orientation='bottom')
            w.sizeHint = lambda: QtCore.QSize(300, 35)
            w.sigChanged = w.sigGradientChangeFinished
            w.sigChanging = w.sigGradientChanged
            w.value = w.colorMap
            w.setValue = w.setColorMap
            self.hideWidget = False
            self.asSubItem = True
        else:
            raise Exception("Unknown type '%s'" % asUnicode(t))
        return w
        
    def widgetEventFilter(self, obj, ev):
        ## filter widget's events
        ## catch TAB to change focus
        ## catch focusOut to hide editor
        if ev.type() == ev.Type.KeyPress:
            if ev.key() == QtCore.Qt.Key.Key_Tab:
                self.focusNext(forward=True)
                return True ## don't let anyone else see this event
            elif ev.key() == QtCore.Qt.Key.Key_Backtab:
                self.focusNext(forward=False)
                return True ## don't let anyone else see this event
            
        return False
        
    def setFocus(self):
        self.showEditor()
        
    def isFocusable(self):
        return self.param.opts['visible'] and self.param.opts['enabled'] and self.param.writable()

    def valueChanged(self, param, val, force=False):
        ## called when the parameter's value has changed
        ParameterItem.valueChanged(self, param, val)
        if force or not fn.eq(val, self.widget.value()):
            try:
                self.widget.sigChanged.disconnect(self.widgetValueChanged)
                self.param.sigValueChanged.disconnect(self.valueChanged)
                self.widget.setValue(val)
                self.param.setValue(self.widget.value())
            finally:
                self.widget.sigChanged.connect(self.widgetValueChanged)
                self.param.sigValueChanged.connect(self.valueChanged)
        self.updateDisplayLabel()  ## always make sure label is updated, even if values match!
        self.updateDefaultBtn()
        
    def updateDefaultBtn(self):
        ## enable/disable default btn 
        self.defaultBtn.setEnabled(
            not self.param.valueIsDefault() and self.param.opts['enabled'] and self.param.writable())
        
        # hide / show
        self.defaultBtn.setVisible(self.param.hasDefault() and not self.param.readonly())

    def updateDisplayLabel(self, value=None):
        """Update the display label to reflect the value of the parameter."""
        if value is None:
            value = self.param.value()
        opts = self.param.opts
        if isinstance(self.widget, QtGui.QAbstractSpinBox):
            text = asUnicode(self.widget.lineEdit().text())
        elif isinstance(self.widget, QtGui.QComboBox):
            text = self.widget.currentText()
        else:
            text = asUnicode(value)
        self.displayLabel.setText(text)

    def widgetValueChanged(self):
        ## called when the widget's value has been changed by the user
        val = self.widget.value()
        newVal = self.param.setValue(val)

    def widgetValueChanging(self, *args):
        """
        Called when the widget's value is changing, but not finalized.
        For example: editing text before pressing enter or changing focus.
        """
        self.param.sigValueChanging.emit(self.param, self.widget.value())
        
    def selected(self, sel):
        """Called when this item has been selected (sel=True) OR deselected (sel=False)"""
        ParameterItem.selected(self, sel)
        
        if self.widget is None:
            return
        if sel and self.param.writable():
            self.showEditor()
        elif self.hideWidget:
            self.hideEditor()

    def showEditor(self):
        self.widget.show()
        self.displayLabel.hide()
        self.widget.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
        if isinstance(self.widget, SpinBox):
            self.widget.selectNumber()  # select the numerical portion of the text for quick editing

    def hideEditor(self):
        self.widget.hide()
        self.displayLabel.show()

    def limitsChanged(self, param, limits):
        """Called when the parameter's limits have changed"""
        ParameterItem.limitsChanged(self, param, limits)
        
        t = self.param.opts['type']
        if t == 'int' or t == 'float':
            self.widget.setOpts(bounds=limits)
        else:
            return  ## don't know what to do with any other types..

    def defaultChanged(self, param, value):
        self.updateDefaultBtn()

    def treeWidgetChanged(self):
        """Called when this item is added or removed from a tree."""
        ParameterItem.treeWidgetChanged(self)
        
        ## add all widgets for this item into the tree
        if self.widget is not None:
            tree = self.treeWidget()
            if tree is None:
                return
            if self.asSubItem:
                self.subItem.setFirstColumnSpanned(True)
                tree.setItemWidget(self.subItem, 0, self.widget)
            tree.setItemWidget(self, 1, self.layoutWidget)
            self.displayLabel.hide()
            self.selected(False)

    def defaultClicked(self):
        self.param.setToDefault()

    def optsChanged(self, param, opts):
        """Called when any options are changed that are not
        name, value, default, or limits"""
        ParameterItem.optsChanged(self, param, opts)

        if 'enabled' in opts:
            self.updateDefaultBtn()
            self.widget.setEnabled(opts['enabled'])

        if 'readonly' in opts:
            self.updateDefaultBtn()
            if hasattr(self.widget, 'setReadOnly'):
                self.widget.setReadOnly(opts['readonly'])
            else:
                self.widget.setEnabled(self.param.opts['enabled'] and not opts['readonly'])

        if 'tip' in opts:
            self.widget.setToolTip(opts['tip'])
        
        ## If widget is a SpinBox, pass options straight through
        if isinstance(self.widget, SpinBox):
            # send only options supported by spinbox
            sbOpts = {}
            if 'units' in opts and 'suffix' not in opts:
                sbOpts['suffix'] = opts['units']
            for k,v in opts.items():
                if k in self.widget.opts:
                    sbOpts[k] = v
            self.widget.setOpts(**sbOpts)
            self.updateDisplayLabel()
        
            
class EventProxy(QtCore.QObject):
    def __init__(self, qobj, callback):
        QtCore.QObject.__init__(self)
        self.callback = callback
        qobj.installEventFilter(self)
        
    def eventFilter(self, obj, ev):
        return self.callback(obj, ev)


class SimpleParameter(Parameter):
    """Parameter representing a single value.

    This parameter is backed by :class:`WidgetParameterItem` to represent the
    following parameter names:

      - 'int'
      - 'float'
      - 'bool'
      - 'str'
      - 'color'
      - 'colormap'
    """
    itemClass = WidgetParameterItem

    def __init__(self, *args, **kargs):
        """Initialize the parameter.

        This is normally called implicitly through :meth:`Parameter.create`.
        The keyword arguments avaialble to :meth:`Parameter.__init__` are
        applicable.
        """
        Parameter.__init__(self, *args, **kargs)
        
        ## override a few methods for color parameters
        if self.opts['type'] == 'color':
            self.value = self.colorValue
            self.saveState = self.saveColorState

    def colorValue(self):
        return fn.mkColor(Parameter.value(self))
    
    def saveColorState(self, *args, **kwds):
        state = Parameter.saveState(self, *args, **kwds)
        state['value'] = fn.colorTuple(self.value())
        return state
        
    def _interpretValue(self, v):
        fn = {
            'int': int,
            'float': float,
            'bool': bool,
            'str': asUnicode,
            'color': self._interpColor,
            'colormap': self._interpColormap,
        }[self.opts['type']]
        return fn(v)
    
    def _interpColor(self, v):
        return fn.mkColor(v)
    
    def _interpColormap(self, v):
        if not isinstance(v, ColorMap):
            raise TypeError("Cannot set colormap parameter from object %r" % v)
        return v


registerParameterType('int', SimpleParameter, override=True)
registerParameterType('float', SimpleParameter, override=True)
registerParameterType('bool', SimpleParameter, override=True)
registerParameterType('str', SimpleParameter, override=True)
registerParameterType('color', SimpleParameter, override=True)
registerParameterType('colormap', SimpleParameter, override=True)


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
                self.addWidget = QtGui.QComboBox()
                self.addWidget.setSizeAdjustPolicy(QtGui.QComboBox.SizeAdjustPolicy.AdjustToContents)
                self.updateAddList()
                self.addWidget.currentIndexChanged.connect(self.addChanged)
            else:
                self.addWidget = QtGui.QPushButton(addText)
                self.addWidget.clicked.connect(self.addClicked)
            w = QtGui.QWidget()
            l = QtGui.QHBoxLayout()
            l.setContentsMargins(0,0,0,0)
            w.setLayout(l)
            l.addWidget(self.addWidget)
            l.addStretch()
            self.addWidgetBox = w
            self.addItem = QtGui.QTreeWidgetItem([])
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
        typ = asUnicode(self.addWidget.currentText())
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

    def addNew(self, typ=None):
        """
        This method is called when the user has requested to add a new item to the group.
        By default, it emits ``sigAddNew(self, typ)``.
        """
        self.sigAddNew.emit(self, typ)
    
    def setAddList(self, vals):
        """Change the list of options available for the user to add to the group."""
        self.setOpts(addList=vals)


registerParameterType('group', GroupParameter, override=True)


class ListParameterItem(WidgetParameterItem):
    """
    WidgetParameterItem subclass providing comboBox that lets the user select from a list of options.
    
    """
    def __init__(self, param, depth):
        self.targetValue = None
        WidgetParameterItem.__init__(self, param, depth)
        
    def makeWidget(self):
        opts = self.param.opts
        t = opts['type']
        w = QtGui.QComboBox()
        w.setMaximumHeight(20)  ## set to match height of spin box and line edit
        w.sigChanged = w.currentIndexChanged
        w.value = self.value
        w.setValue = self.setValue
        self.widget = w  ## needs to be set before limits are changed
        self.limitsChanged(self.param, self.param.opts['limits'])
        if len(self.forward) > 0:
            self.setValue(self.param.value())
        return w
        
    def value(self):
        key = asUnicode(self.widget.currentText())
        
        return self.forward.get(key, None)
            
    def setValue(self, val):
        self.targetValue = val
        if val not in self.reverse[0]:
            self.widget.setCurrentIndex(0)
        else:
            key = self.reverse[1][self.reverse[0].index(val)]
            ind = self.widget.findText(key)
            self.widget.setCurrentIndex(ind)

    def limitsChanged(self, param, limits):
        # set up forward / reverse mappings for name:value
        
        if len(limits) == 0:
            limits = ['']  ## Can never have an empty list--there is always at least a singhe blank item.
        
        self.forward, self.reverse = ListParameter.mapping(limits)
        try:
            self.widget.blockSignals(True)
            val = self.targetValue  #asUnicode(self.widget.currentText())
            
            self.widget.clear()
            for k in self.forward:
                self.widget.addItem(k)
                if k == val:
                    self.widget.setCurrentIndex(self.widget.count()-1)
                    self.updateDisplayLabel()
        finally:
            self.widget.blockSignals(False)


class ListParameter(Parameter):
    """Parameter with a list of acceptable values.

    By default, this parameter is represtented by a :class:`ListParameterItem`,
    displaying a combo box to select a value from the list.

    In addition to the generic :class:`~pyqtgraph.parametertree.Parameter`
    options, this parameter type accepts a ``limits`` argument specifying the
    list of allowed values.  ``values`` is an alias and may be used instead.

    The values may generally be of any data type, as long as they can be
    represented as a string. If the string representation provided is
    undesirable, the values may be given as a dictionary mapping the desired
    string representation to the value.
    """

    itemClass = ListParameterItem

    def __init__(self, **opts):
        self.forward = OrderedDict()  ## {name: value, ...}
        self.reverse = ([], [])       ## ([value, ...], [name, ...])
        
        # Parameter uses 'limits' option to define the set of allowed values
        if 'values' in opts:
            opts['limits'] = opts['values']
        if opts.get('limits', None) is None:
            opts['limits'] = []
        Parameter.__init__(self, **opts)
        self.setLimits(opts['limits'])
        
    def setLimits(self, limits):
        """Change the list of allowed values."""
        self.forward, self.reverse = self.mapping(limits)
        
        Parameter.setLimits(self, limits)
        if len(self.reverse[0]) > 0 and self.value() not in self.reverse[0]:
            self.setValue(self.reverse[0][0])
            
    @staticmethod
    def mapping(limits):
        # Return forward and reverse mapping objects given a limit specification
        forward = OrderedDict()  ## {name: value, ...}
        reverse = ([], [])       ## ([value, ...], [name, ...])
        if isinstance(limits, dict):
            for k, v in limits.items():
                forward[k] = v
                reverse[0].append(v)
                reverse[1].append(k)
        else:
            for v in limits:
                n = asUnicode(v)
                forward[n] = v
                reverse[0].append(v)
                reverse[1].append(n)
        return forward, reverse

registerParameterType('list', ListParameter, override=True)



class ActionParameterItem(ParameterItem):
    """ParameterItem displaying a clickable button."""
    def __init__(self, param, depth):
        ParameterItem.__init__(self, param, depth)
        self.layoutWidget = QtGui.QWidget()
        self.layout = QtGui.QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layoutWidget.setLayout(self.layout)
        self.button = QtGui.QPushButton()
        #self.layout.addSpacing(100)
        self.layout.addWidget(self.button)
        self.layout.addStretch()
        self.button.clicked.connect(self.buttonClicked)
        self.titleChanged()
        self.optsChanged(self.param, self.param.opts)

    def treeWidgetChanged(self):
        ParameterItem.treeWidgetChanged(self)
        tree = self.treeWidget()
        if tree is None:
            return
        
        self.setFirstColumnSpanned(True)
        tree.setItemWidget(self, 0, self.layoutWidget)

    def titleChanged(self):
        self.button.setText(self.param.title())
        self.setSizeHint(0, self.button.sizeHint())

    def optsChanged(self, param, opts):
        ParameterItem.optsChanged(self, param, opts)

        if 'enabled' in opts:
            self.button.setEnabled(opts['enabled'])

        if 'tip' in opts:
            self.button.setToolTip(opts['tip'])

    def buttonClicked(self):
        self.param.activate()
        
class ActionParameter(Parameter):
    """Used for displaying a button within the tree.

    ``sigActivated(self)`` is emitted when the button is clicked.
    """
    itemClass = ActionParameterItem
    sigActivated = QtCore.Signal(object)
    
    def activate(self):
        self.sigActivated.emit(self)
        self.emitStateChanged('activated', None)
        
registerParameterType('action', ActionParameter, override=True)


class TextParameterItem(WidgetParameterItem):
    """ParameterItem displaying a QTextEdit widget."""
    
    def makeWidget(self):
        self.hideWidget = False
        self.asSubItem = True
        self.textBox = w = QtGui.QTextEdit()
        w.sizeHint = lambda: QtCore.QSize(300, 100)
        w.value = lambda: str(w.toPlainText())
        w.setValue = w.setPlainText
        w.sigChanged = w.textChanged
        return w


class TextParameter(Parameter):
    """Editable string, displayed as large text box in the tree."""
    itemClass = TextParameterItem


registerParameterType('text', TextParameter, override=True)

class FileDialogItem(ParameterItem):

    def __init__(self, param, depth):
        ParameterItem.__init__(self, param, depth)

        self.dialogTypes={
        "getExistingDirectory" : (QtGui.QFileDialog().getExistingDirectory,"Open directory"),
        #"getExistingDirectoryUrl" : (QtGui.QFileDialog().getExistingDirectoryUrl,"Open directory"),
        "getOpenFileName" : (QtGui.QFileDialog().getOpenFileName,"Open file"),
        "getOpenFileNames" : (QtGui.QFileDialog().getOpenFileNames,"Open files"),
        #"getOpenFileUrl" : (QtGui.QFileDialog().getOpenFileUrl,"Open directory"),
        #"getOpenFileUrls" : (QtGui.QFileDialog().getOpenFileUrls,"Open directory"),
        "getSaveFileName" : (QtGui.QFileDialog().getSaveFileName,"Save file"),
        #"getSaveFileUrl" : (QtGui.QFileDialog().getSaveFileUrl,"Open directory"),
        "openFiles" : (QtGui.QFileDialog().getOpenFileNames,"Open files"),
        "openFile" : (QtGui.QFileDialog().getOpenFileName,"Open file"),
        "openDirectory" : (QtGui.QFileDialog().getExistingDirectory,"Open directory"),
        "saveFile" : (QtGui.QFileDialog().getSaveFileName,"Save file")}

        self.layoutWidget = QtGui.QWidget()
        self.layout = QtGui.QHBoxLayout()
        self.layoutWidget.setLayout(self.layout)
        self.button = QtGui.QPushButton(param.name())
        self.label = QtGui.QLabel()
        #self.layout.addSpacing(100)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.label)
        self.layout.addStretch()
        self.setText(0, '')

        self.returnedValues = None
        self.lastDirectory = os.path.dirname(os.path.abspath(__file__))
        self.button.clicked.connect(self.buttonClicked)

    def treeWidgetChanged(self):
        ParameterItem.treeWidgetChanged(self)
        tree = self.treeWidget()
        if tree is None:
            return

        tree.resizeColumnToContents(0)
        tree.setItemWidget(self, 0, self.layoutWidget)

    def buttonClicked(self):
        filterString = self.param.opts.get('filterString', "All files (*.*)")
        dialogType = self.param.opts.get('dialogType', "getOpenFileName")
        dialogFunction, dialogCaption = self.dialogTypes[dialogType]
        dialogCaption = self.param.opts.get('dialogCaption', dialogCaption)

        if dialogType in ["getExistingDirectory","openDirectory"]:
            fn = dialogFunction(None, dialogCaption, self.lastDirectory)
            fn = os.path.realpath(asUnicode(fn))
            if fn=="": return 0
            self.lastDirectory = os.path.dirname(fn)
            self.label.setText(str(fn))
            tooltip = str(os.path.realpath(fn))
        elif dialogType in ["getOpenFileNames","openFiles"]:
            fn = dialogFunction(None, dialogCaption, self.lastDirectory, filterString)
            fn = [os.path.realpath(asUnicode(f)) for f in fn if os.path.isfile(f) and f != ""]
            if len(fn)<=0: return 0
            self.lastDirectory = os.path.dirname(fn[0])
            self.label.setText(str(fn[0]))
            tooltip = "\n".join(fn)
        else:
            fn = dialogFunction(None, dialogCaption, self.lastDirectory, filterString)
            if fn == "": return 0
            fn = os.path.realpath(asUnicode(fn))
            self.lastDirectory = os.path.dirname(fn)
            self.label.setText(str(fn))
            tooltip = str(fn)

        self.label.setToolTip(tooltip)
        self.param.fileSelected(fn)

class FileDialogParameter(Parameter):
    """Used for displaying a button within the tree."""
    itemClass = FileDialogItem
    sigFileSelected = QtCore.Signal(object,object)

    def fileSelected(self,f):
        self.sigFileSelected.emit(self,f)
        self.emitStateChanged('fileSelected', f)

registerParameterType('file', FileDialogParameter, override=True)

class ProgressBarParameterItem(WidgetParameterItem):
    def makeWidget(self):
        w = QtGui.QProgressBar()
        w.setMaximumHeight(20)
        w.sigChanged = w.valueChanged
        self.widget = w
        self.hideWidget = False
        return w

class ProgressBarParameter(Parameter):
    itemClass = ProgressBarParameterItem

registerParameterType('progress', ProgressBarParameter, override=True)

# WidgetParameterItem is not a QObject, and the slider's value needs to be converted before
# emitting. So, create an emitter class here that can be used instead
class Emitter(QtCore.QObject):
    sigChanging = QtCore.Signal(object, object)

class SliderParameterItem(WidgetParameterItem):
    slider: QtGui.QSlider
    span: np.ndarray
    charSpan: np.ndarray

    def __init__(self, param, depth):
        # Bind emitter to self to avoid garbage collection
        self.emitter = Emitter()
        self.sigChanging = self.emitter.sigChanging
        self._suffix = None
        super().__init__(param, depth)

    def optsChanged(self, param, opts):
        if 'suffix' in opts:
            self.setSuffix(opts['suffix'])
            self.slider.valueChanged.emit(self.slider.value())
        super().optsChanged(param, opts)

    def updateDisplayLabel(self, value=None):
        if value is None:
            value = self.param.value()
        value = asUnicode(value)
        if self._suffix is None:
            suffixTxt = ''
        else:
            suffixTxt = f' {self._suffix}'
        self.displayLabel.setText(value + suffixTxt)

    def setSuffix(self, suffix):
        self._suffix = suffix
        self._updateLabel(self.slider.value())

    def makeWidget(self):
        param = self.param
        opts = param.opts
        self._suffix = opts.get('suffix')

        self.slider = QtGui.QSlider()
        self.slider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        lbl = QtGui.QLabel()
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        w = QtGui.QWidget()
        layout = QtGui.QHBoxLayout()
        w.setLayout(layout)
        layout.addWidget(lbl)
        layout.addWidget(self.slider)

        def setValue(v):
            self.slider.setValue(self.spanToSliderValue(v))
        def getValue():
            return self.span[self.slider.value()].item()

        def vChanged(v):
            lbl.setText(self.prettyTextValue(v))
        self.slider.valueChanged.connect(vChanged)

        def onMove(pos):
            self.sigChanging.emit(self, self.span[pos].item())
        self.slider.sliderMoved.connect(onMove)

        w.setValue = setValue
        w.value = getValue
        w.sigChanged = self.slider.valueChanged
        w.sigChanging = self.sigChanging
        self.optsChanged(param, opts)
        return w

    # def updateDisplayLabel(self, value=None):
    #   self.displayLabel.setText(self.prettyTextValue(value))

    def spanToSliderValue(self, v):
        return int(np.argmin(np.abs(self.span-v)))

    def prettyTextValue(self, v):
        if self._suffix is None:
            suffixTxt = ''
        else:
            suffixTxt = f' {self._suffix}'
        format_ = self.param.opts.get('format', None)
        cspan = self.charSpan
        if format_ is None:
            format_ = f'{{0:>{cspan.dtype.itemsize}}}{suffixTxt}'
        return format_.format(cspan[v].decode())

    def optsChanged(self, param, opts):
        try:
            super().optsChanged(param, opts)
        except AttributeError as ex:
            pass
        span = opts.get('span', None)
        if span is None:
            step = opts.get('step', 1)
            start, stop = opts['limits']
            # Add a bit to 'stop' since python slicing excludes the last value
            span = np.arange(start, stop+step, step)
        precision = opts.get('precision', 2)
        if precision is not None:
            span = span.round(precision)
        self.span = span
        self.charSpan = np.char.array(span)
        w = self.slider
        w.setMinimum(0)
        w.setMaximum(len(span)-1)

    def limitsChanged(self, param, limits):
        self.optsChanged(param, dict(limits=limits))

class SliderParameter(Parameter):
    itemClass = SliderParameterItem

registerParameterType('slider', SliderParameter, override=True)

class FontParameterItem(WidgetParameterItem):
    def makeWidget(self):
        w = QtGui.QFontComboBox()
        w.setMaximumHeight(20)
        w.sigChanged = w.currentFontChanged
        w.value = w.currentFont
        w.setValue = w.setCurrentFont
        self.widget = w
        self.hideWidget = False
        return w

class FontParameter(Parameter):
    itemClass = FontParameterItem

registerParameterType('font', FontParameter, override=True)

class CalendarParameterItem(WidgetParameterItem):
    def makeWidget(self):
        w = QtGui.QCalendarWidget()
        w.setMaximumHeight(200)
        w.sigChanged = w.selectionChanged
        w.value = w.selectedDate
        w.setValue = w.setSelectedDate
        self.widget = w
        self.hideWidget = False
        self.param.opts['default'] = QtCore.QDate.currentDate()
        return w

class CalendarParameter(Parameter):
    itemClass = CalendarParameterItem

registerParameterType('calendar', CalendarParameter, override=True)



class PenParameterItem(ParameterItem):
    def __init__(self, param, depth):
        ParameterItem.__init__(self, param, depth)
        self.layoutWidget = QtGui.QWidget()
        self.layout = QtGui.QHBoxLayout()
        self.layoutWidget.setLayout(self.layout)
        self.button = QtGui.QPushButton()
        #larger button
        self.button.setFixedWidth(100)
        self.layout.addWidget(self.button)
        self.layout.addStretch()
        self.pen = QtGui.QPen()
        self.button.clicked.connect(self.buttonClicked)
        #clear button for drawing, override paint event
        self.setText(0, '')
        self.button.paintEvent = self.buttonPaintEvent
        

    def value(self):
        return self.pen

    def setValue(self,pen):
        self.penChanged(pen)

    def treeWidgetChanged(self):
        ParameterItem.treeWidgetChanged(self)
        tree = self.treeWidget()
        if tree is None:
            return

        tree.resizeColumnToContents(0)
        tree.setItemWidget(self, 0, self.layoutWidget)

    def buttonClicked(self):
        #open up the pen selector dialog
        self.oldPen = fn.mkPen(self.param.value())
        self.pdialog = PenSelectorDialog(fn.mkPen(self.pen),QtGui.QApplication.activeWindow())
        self.pdialog.penChanged.connect(self.penChanged)
        self.pdialog.finished.connect(self.penChangeFinished)
        self.pdialog.exec()

    def penChangeFinished(self,ret):
        #finished changing
        if not ret:
            #revert if cancel
            self.penChanged(self.oldPen)
        else:
            #event if accepted
            self.param.penChanged(self.pen)

    def penChanged(self,pen):
        if not isinstance(pen,QtGui.QPen):
            pen = fn.mkPen(pen)
        pen.setCosmetic(True)
        self.pen = pen

    def buttonPaintEvent(self, event):
        #draw a button as usual
        QtGui.QPushButton.paintEvent(self.button, event)

        path = QtGui.QPainterPath()
        displaySize = self.button.size()
        w,h = displaySize.width(),displaySize.height()
        #draw a squiggle with the pen
        path.moveTo(w*.2,h*.2)
        path.lineTo(w*.4,h*.8)
        path.cubicTo(w*.5,h*.1,w*.7,h*.1,w*.8,h*.8)

        painter = QtGui.QPainter(self.button)
        painter.setPen(self.pen)
        painter.drawPath(path)
        painter.end()

class PenParameter(Parameter):
    itemClass = PenParameterItem
    sigPenChanged = QtCore.Signal(object,object)

    def penChanged(self,pen):
        self.opts['value'] = pen
        self.sigPenChanged.emit(self, pen)
        self.emitStateChanged('penChanged', pen)

registerParameterType('pen', PenParameter, override=True)
