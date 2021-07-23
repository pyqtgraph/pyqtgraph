from ..Qt import QtCore, QtGui, QtWidgets, QT_LIB
from ..python2_3 import asUnicode
from .Parameter import Parameter, registerParameterType
from .ParameterItem import ParameterItem
from ..widgets.SpinBox import SpinBox
from ..widgets.ColorButton import ColorButton
from ..widgets.PenSelectorDialog import PenSelectorDialog
from ..colormap import ColorMap
from .. import icons as icons
from .. import functions as fn
from collections import OrderedDict
import re
import numpy as np
import os


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
            self.subItem = QtWidgets.QTreeWidgetItem()
            self.subItem.depth = self.depth + 1
            self.subItem.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
            self.addChild(self.subItem)

        self.defaultBtn = self.makeDefaultButton()

        self.displayLabel = QtWidgets.QLabel()

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        if not self.asSubItem:
            layout.addWidget(w, 1)
        layout.addWidget(self.displayLabel, 1)
        layout.addStretch(0)
        layout.addWidget(self.defaultBtn)
        self.layoutWidget = QtWidgets.QWidget()
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
            w = QtWidgets.QCheckBox()
            w.sigChanged = w.toggled
            w.value = w.isChecked
            w.setValue = w.setChecked
            self.hideWidget = False
        elif t == 'str':
            w = QtWidgets.QLineEdit()
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

    def makeDefaultButton(self):
        defaultBtn = QtWidgets.QPushButton()
        defaultBtn.setAutoDefault(False)
        defaultBtn.setFixedWidth(20)
        defaultBtn.setFixedHeight(20)
        defaultBtn.setIcon(icons.getGraphIcon('default'))
        defaultBtn.clicked.connect(self.defaultClicked)
        return defaultBtn

    def setFocus(self):
        self.showEditor()

    def isFocusable(self):
        return self.param.opts['visible'] and self.param.opts['enabled'] and self.param.writable()

    def valueChanged(self, param, val, force=False):
        ## called when the parameter's value has changed
        ParameterItem.valueChanged(self, param, val)
        if force or not fn.eq(val, self.widget.value()):
            try:
                if self.widget.sigChanged is not None:
                    self.widget.sigChanged.disconnect(self.widgetValueChanged)
                self.param.sigValueChanged.disconnect(self.valueChanged)
                self.widget.setValue(val)
                self.param.setValue(self.widget.value())
            finally:
                if self.widget.sigChanged is not None:
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
        if isinstance(self.widget, QtWidgets.QAbstractSpinBox):
            text = asUnicode(self.widget.lineEdit().text())
        elif isinstance(self.widget, QtWidgets.QComboBox):
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
        w = QtWidgets.QComboBox()
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
        self.layoutWidget = QtWidgets.QWidget()
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layoutWidget.setLayout(self.layout)
        self.button = QtWidgets.QPushButton()
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
        self.textBox = w = QtWidgets.QTextEdit()
        w.sizeHint = lambda: QtCore.QSize(300, 100)
        w.value = lambda: str(w.toPlainText())
        w.setValue = w.setPlainText
        w.sigChanged = w.textChanged
        return w


class TextParameter(Parameter):
    """Editable string, displayed as large text box in the tree."""
    itemClass = TextParameterItem


registerParameterType('text', TextParameter, override=True)


class Emitter(QtCore.QObject):
    """
    WidgetParameterItem is not a QObject, and the slider's value needs to be converted before
    emitting. So, create an emitter class here that can be used instead
    """
    sigChanging = QtCore.Signal(object, object)
    sigChanged = QtCore.Signal(object, object)


def _set_filepicker_kwargs(fileDlg, **kwargs):
  """Applies a dict of enum/flag kwarg opts to a file dialog"""
  NO_MATCH = object()

  for kk, vv in kwargs.items():
    # Convert string or list representations into true flags
    # 'fileMode' -> 'FileMode'
    formattedName = kk[0].upper() + kk[1:]
    # Edge case: "Options" has enum "Option"
    if formattedName == 'Options':
      enumCls = fileDlg.Option
    else:
      enumCls = getattr(fileDlg, formattedName, NO_MATCH)
    setFunc = getattr(fileDlg, f'set{formattedName}', NO_MATCH)
    if enumCls is NO_MATCH or setFunc is NO_MATCH:
      continue
    if enumCls is fileDlg.Option:
      builder = fileDlg.Option(0)
      # This is the only flag enum, all others can only take one value
      if isinstance(vv, str): vv = [vv]
      for flag in vv:
        curVal = getattr(enumCls, flag)
        builder |= curVal
      # Some Qt implementations turn into ints by this point
      outEnum = enumCls(builder)
    else:
      outEnum = getattr(enumCls, vv)
    setFunc(outEnum)

def popupFilePicker(parent=None, windowTitle='', nameFilter='', directory=None, selectFile=None, relativeTo=None, **kwargs):
    """
    Thin wrapper around Qt file picker dialog. Used internally so all options are consistent
    among all requests for external file information

    ============== ========================================================
    **Arguments:**
    parent         Dialog parent
    windowTitle    Title of dialog window
    nameFilter     File filter as required by the Qt dialog
    directory      Where in the file system to open this dialog
    selectFile     File to preselect
    relativeTo     Parent directory that, if provided, will be removed from the prefix of all returned paths. So,
                   if '/my/text/file.txt' was selected, and `relativeTo='/my/text/'`, the return value would be
                   'file.txt'. This uses os.path.relpath under the hood, so expect that behavior.
    kwargs         Any enum value accepted by a QFileDialog and its value. Values can be a string or list of strings,
                   i.e. fileMode='AnyFile', options=['ShowDirsOnly', 'DontResolveSymlinks'], acceptMode='AcceptSave'
    ============== ========================================================

    """
    fileDlg = QtWidgets.QFileDialog(parent)
    _set_filepicker_kwargs(fileDlg, **kwargs)

    fileDlg.setModal(True)
    if directory is not None:
        fileDlg.setDirectory(directory)
    fileDlg.setNameFilter(nameFilter)
    if selectFile is not None:
        fileDlg.selectFile(selectFile)

    fileDlg.setWindowTitle(windowTitle)

    if fileDlg.exec():
        # Append filter type
        singleExtReg = r'(\.\w+)'
        # Extensions of type 'myfile.ext.is.multi.part' need to capture repeating pattern of singleExt
        suffMatch = re.search(rf'({singleExtReg}+)', fileDlg.selectedNameFilter())
        if suffMatch:
            # Strip leading '.' if it exists
            ext = suffMatch.group(1)
            if ext.startswith('.'):
                ext = ext[1:]
            fileDlg.setDefaultSuffix(ext)
        fList = fileDlg.selectedFiles()
    else:
        fList = []
    if relativeTo is not None:
        fList = [os.path.relpath(file, relativeTo) for file in fList]
    # Make consistent to os flavor
    fList = [os.path.normpath(file) for file in fList]
    if fileDlg.fileMode() == fileDlg.FileMode.ExistingFiles:
        return fList
    elif len(fList) > 0:
        return fList[0]
    else:
        return None

# class FileParameterItem(WidgetParameterItem):
#     def __init__(self, param, depth):
#         self._value = None
#         # Temporarily consider string during construction
#         oldType = param.opts.get('type')
#         param.opts['type'] = 'str'
#         super().__init__(param, depth)
#         param.opts['type'] = oldType
#
#         button = QtWidgets.QPushButton('...')
#         button.setFixedWidth(25)
#         button.setContentsMargins(0, 0, 0, 0)
#         button.clicked.connect(self._retrieveFileSelection_gui)
#         self.layoutWidget.layout().insertWidget(2, button)
#         self.displayLabel.resizeEvent = self._newResizeEvent
#         # self.layoutWidget.layout().insertWidget(3, self.defaultBtn)
#
#     def makeWidget(self):
#         w = super().makeWidget()
#         w.setValue = self.setValue
#         w.value = self.value
#         # Doesn't make much sense to have a 'changing' signal since filepaths should be complete before value
#         # is emitted
#         delattr(w, 'sigChanging')
#         return w
#
#     def _newResizeEvent(self, ev):
#         ret = type(self.displayLabel).resizeEvent(self.displayLabel, ev)
#         self.updateDisplayLabel()
#         return ret
#
#     def setValue(self, value):
#         self._value = value
#         self.widget.setText(asUnicode(value))
#
#     def value(self):
#         return self._value
#
#     def _retrieveFileSelection_gui(self):
#         curVal = self.param.value()
#         if isinstance(curVal, list) and len(curVal):
#             # All files should be from the same directory, in principle
#             # Since no mechanism exists for preselecting multiple, the most sensible
#             # thing is to select nothing in the preview dialog
#             curVal = curVal[0]
#             if os.path.isfile(curVal):
#                 curVal = os.path.dirname(curVal)
#         opts = self.param.opts.copy()
#         useDir = curVal or opts.get('directory') or os.getcwd()
#         startDir = os.path.abspath(useDir)
#         if os.path.isfile(startDir):
#             opts['selectFile'] = os.path.basename(startDir)
#             startDir = os.path.dirname(startDir)
#         if os.path.exists(startDir):
#             opts['directory'] = startDir
#         opts.setdefault('windowTitle', self.param.title())
#
#         fname = popupFilePicker(None, **opts)
#         if not fname:
#             return
#         self.param.setValue(fname)
#
#     def updateDefaultBtn(self):
#         # Override since a readonly label should still allow reverting to default
#         ## enable/disable default btn
#         self.defaultBtn.setEnabled(
#             not self.param.valueIsDefault() and self.param.opts['enabled'])
#
#         # hide / show
#         self.defaultBtn.setVisible(self.param.hasDefault())
#
#     def updateDisplayLabel(self, value=None):
#         lbl = self.displayLabel
#         if value is None:
#             value = self.param.value()
#         value = asUnicode(value)
#         font = lbl.font()
#         metrics = QtGui.QFontMetricsF(font)
#         value = metrics.elidedText(value, QtCore.Qt.TextElideMode.ElideLeft, lbl.width()-5)
#         return super().updateDisplayLabel(value)

# class FileParameter(Parameter):
#     """
#     Interfaces with the myriad of file options available from a QFileDialog.
#
#     Note that the output can either be a single file string or list of files, depending on whether
#     `fileMode='ExistingFiles'` is specified.
#
#     Note that in all cases, absolute file paths are returned unless `relativeTo` is specified as
#     elaborated below.
#
#     ============== ========================================================
#     **Options:**
#     parent         Dialog parent
#     winTitle       Title of dialog window
#     nameFilter     File filter as required by the Qt dialog
#     directory      Where in the file system to open this dialog
#     selectFile     File to preselect
#     relativeTo     Parent directory that, if provided, will be removed from the prefix of all returned paths. So,
#                    if '/my/text/file.txt' was selected, and `relativeTo='my/text/'`, the return value would be
#                    'file.txt'. This uses os.path.relpath under the hood, so expect that behavior.
#     kwargs         Any enum value accepted by a QFileDialog and its value. Values can be a string or list of strings,
#                    i.e. fileMode='AnyFile', options=['ShowDirsOnly', 'DontResolveSymlinks']
#     ============== ========================================================
#     """
#     itemClass = FileParameterItem
#
#     def __init__(self, **opts):
#         opts.setdefault('readonly', True)
#         super().__init__(**opts)


# class ProgressBarParameterItem(WidgetParameterItem):
#     def makeWidget(self):
#         w = QtWidgets.QProgressBar()
#         w.setMaximumHeight(20)
#         w.sigChanged = w.valueChanged
#         self.hideWidget = False
#         return w
#
# class ProgressBarParameter(Parameter):
#     """
#     Displays a progress bar whose value can be set between 0 and 100
#     """
#     itemClass = ProgressBarParameterItem

class SliderParameterItem(WidgetParameterItem):
    slider: QtWidgets.QSlider
    span: np.ndarray
    charSpan: np.ndarray

    def __init__(self, param, depth):
        # Bind emitter to self to avoid garbage collection
        self.emitter = Emitter()
        self.sigChanging = self.emitter.sigChanging
        self._suffix = None
        super().__init__(param, depth)

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

        self.slider = QtWidgets.QSlider()
        self.slider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        lbl = QtWidgets.QLabel()
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        w = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
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
        if 'suffix' in opts:
            self.setSuffix(opts['suffix'])
            self.slider.valueChanged.emit(self.slider.value())

    def limitsChanged(self, param, limits):
        self.optsChanged(param, dict(limits=limits))

class SliderParameter(Parameter):
    """
    ============== ========================================================
    **Options**
    limits         [start, stop] numbers
    step:          Defaults to 1, the spacing between each slider tick
    span:          Instead of limits + step, span can be set to specify
                   the range of slider options (e.g. np.linspace(-pi, pi, 100))
    format:        Format string to determine number of decimals to show, etc.
                   Defaults to display based on span dtype
    precision:     int number of decimals to keep for float tick spaces
    ============== ========================================================
    """
    itemClass = SliderParameterItem

# class FontParameterItem(WidgetParameterItem):
#     def makeWidget(self):
#         w = QtWidgets.QFontComboBox()
#         w.setMaximumHeight(20)
#         w.sigChanged = w.currentFontChanged
#         w.value = w.currentFont
#         w.setValue = w.setCurrentFont
#         self.widget = w
#         self.hideWidget = False
#         return w

# class FontParameter(Parameter):
#     """
#     Creates and controls a QFont value. Be careful when selecting options from the font dropdown. since not all
#     fonts are available on all systems
#     """
#     itemClass = FontParameterItem

#     def _interpretValue(self, v):
#         if isinstance(v, str):
#             newVal = QtGui.QFont()
#             if not newVal.fromString(v):
#                 raise ValueError(f'Error parsing font "{v}"')
#             v = newVal
#         return v

#     def saveState(self, filter=None):
#         state = super().saveState(filter)
#         state['value'] = state['value'].toString()
#         return state

# class CalendarParameterItem(WidgetParameterItem):
#     def makeWidget(self):
#         self.asSubItem = True
#         w = QtWidgets.QCalendarWidget()
#         w.setMaximumHeight(200)
#         w.sigChanged = w.selectionChanged
#         w.value = w.selectedDate
#         w.setValue = w.setSelectedDate
#         self.widget = w
#         self.hideWidget = False
#         self.param.opts.setdefault('default', QtCore.QDate.currentDate())
#         return w

# class CalendarParameter(Parameter):
#     """
#     Displays a Qt calendar whose date is specified by a 'format' option.

#     ============== ========================================================
#     **Options:**
#     format         Format for displaying the date and converting from a string. Can be any value accepted by
#                    `QDate.toString` and `fromString`, or a stringified version of a QDateFormat enum, i.e. 'ISODate',
#                    'TextDate' (default), etc.
#     ============== ========================================================
#     """

#     itemClass = CalendarParameterItem

#     def __init__(self, **opts):
#         opts.setdefault('format', 'TextDate')
#         super().__init__(**opts)

#     def _interpretFormat(self, fmt=None):
#         fmt = fmt or self.opts.get('format')
#         if hasattr(QtCore.Qt.DateFormat, fmt):
#             fmt = getattr(QtCore.Qt.DateFormat, fmt)
#         return fmt

#     def _interpretValue(self, v):
#         if isinstance(v, str):
#             fmt = self._interpretFormat()
#             if fmt is None:
#                 raise ValueError('Cannot parse date string without a set format')
#             v = QtCore.QDate.fromString(v, fmt)
#         return v

#     def saveState(self, filter=None):
#         state = super().saveState(filter)
#         fmt = self._interpretFormat()
#         state['value'] = state['value'].toString(fmt)
#         return state


# class QtEnumParameter(ListParameter):
#     def __init__(self, enum, searchObj=QtCore.Qt, **opts):
#         """
#         Constructs a list of allowed enum values from the enum class provided
#         `searchObj` is only needed for PyQt5 compatibility, where it must be the module holding the enum.
#         For instance, if making a QtEnumParameter out of QtWidgets.QFileDialog.Option, `searchObj` would
#         be QtWidgets.QFileDialog
#         """
#         self.enum = enum
#         self.searchObj = searchObj
#         opts.setdefault('name', enum.__name__)
#         self.enumMap = self._getAllowedEnums(enum)

#         opts.update(limits=self.formattedLimits())
#         super().__init__(**opts)

#     def setValue(self, value, blockSignal=None):
#         if isinstance(value, str):
#             value = self.enumMap[value]
#         super().setValue(value, blockSignal)

#     def formattedLimits(self):
#         # Title-cased words without the ending substring for brevity
#         substringEnd = None
#         mapping = self.enumMap
#         shortestName = min(len(name) for name in mapping)
#         names = list(mapping)
#         cmpName, *names = names
#         for ii in range(-1, -shortestName-1, -1):
#             if any(cmpName[ii] != curName[ii] for curName in names):
#                 substringEnd = ii+1
#                 break
#         # Special case of 0: Set to none to avoid null string
#         if substringEnd == 0:
#             substringEnd = None
#         limits = {}
#         for kk, vv in self.enumMap.items():
#             limits[kk[:substringEnd]] = vv
#         return limits

#     def saveState(self, filter=None):
#         state = super().saveState(filter)
#         reverseMap = dict(zip(self.enumMap.values(), self.enumMap))
#         state['value'] = reverseMap[state['value']]
#         return state

#     def _getAllowedEnums(self, enum):
#         """Pyside provides a dict for easy evaluation"""
#         if 'PySide' in QT_LIB:
#             vals = enum.values
#         elif 'PyQt5' in QT_LIB:
#             vals = {}
#             for key in dir(self.searchObj):
#                 value = getattr(self.searchObj, key)
#                 if isinstance(value, enum):
#                     vals[key] = value
#         elif 'PyQt6' in QT_LIB:
#             vals = {e.name: e for e in enum}
#         else:
#             raise RuntimeError(f'Cannot find associated enum values for qt lib {QT_LIB}')
#         # Remove "M<enum>" since it's not a real option
#         vals.pop(f'M{enum.__name__}', None)
#         return vals

# class PenParameterItem(WidgetParameterItem):
#     def __init__(self, param, depth):
#         self.pdialog = PenSelectorDialog(fn.mkPen(param.pen))
#         self.pdialog.setModal(True)
#         self.pdialog.accepted.connect(self.penChangeFinished)
#         super().__init__(param, depth)
#         self.displayLabel.paintEvent = self.displayPaintEvent

#     def makeWidget(self):
#         self.button = QtWidgets.QPushButton()
#         #larger button
#         self.button.setFixedWidth(100)
#         self.button.clicked.connect(self.buttonClicked)
#         self.button.paintEvent = self.buttonPaintEvent
#         self.button.value = self.value
#         self.button.setValue = self.setValue
#         self.button.sigChanged = None
#         return self.button

#     @property
#     def pen(self):
#         return self.pdialog.pen

#     def value(self):
#         return self.pen

#     def setValue(self, pen):
#         self.pdialog.updateParamFromPen(self.pdialog.param, pen)

#     def updateDisplayLabel(self, value=None):
#         super().updateDisplayLabel('')
#         self.displayLabel.update()
#         self.widget.update()

#     def buttonClicked(self):
#         #open up the pen selector dialog
#         # Copy in case of rejection
#         prePen = QtGui.QPen(self.pen)
#         if self.pdialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
#             self.pdialog.updateParamFromPen(self.pdialog.param, prePen)

#     def penChangeFinished(self):
#         self.param.setValue(self.pdialog.pen)

#     def penPaintEvent(self, event, item):
#         # draw item as usual
#         type(item).paintEvent(item, event)

#         path = QtGui.QPainterPath()
#         displaySize = item.size()
#         w, h = displaySize.width(), displaySize.height()
#         # draw a squiggle with the pen
#         path.moveTo(w * .2, h * .2)
#         path.lineTo(w * .4, h * .8)
#         path.cubicTo(w * .5, h * .1, w * .7, h * .1, w * .8, h * .8)

#         painter = QtGui.QPainter(item)
#         painter.setPen(self.pen)
#         painter.drawPath(path)
#         painter.end()

#     def buttonPaintEvent(self, event):
#         return self.penPaintEvent(event, self.button)

#     def displayPaintEvent(self, event):
#         return self.penPaintEvent(event, self.displayLabel)

# class PenParameter(Parameter):
#     """
#     Controls the appearance of a QPen value.

#     When `saveState` is called, the value is encoded as (color, width, style, capStyle, joinStyle, cosmetic)

#     ============== ========================================================
#     **Options:**
#     color          pen color, can be any argument accepted by :func:`~pyqtgraph.mkColor` (defaults to black)
#     width          integer width >= 0 (defaults to 1)
#     style          String version of QPenStyle enum, i.e. 'SolidLine' (default), 'DashLine', etc.
#     capStyle       String version of QPenCapStyle enum, i.e. 'SquareCap' (default), 'RoundCap', etc.
#     joinStyle      String version of QPenJoinStyle enum, i.e. 'BevelJoin' (default), 'RoundJoin', etc.
#     cosmetic       Boolean, whether or not the pen is cosmetic (defaults to True)
#     ============== ========================================================
#     """

#     itemClass = PenParameterItem
#     sigPenChanged = QtCore.Signal(object,object)

#     def __init__(self, **opts):
#         self.pen = fn.mkPen()
#         self.penOptsParam = PenSelectorDialog.mkParam(self.pen)
#         super().__init__(**opts)

#     def saveState(self, filter=None):
#         state = super().saveState(filter)
#         overrideState = self.penOptsParam.saveState(filter)['children']
#         state['value'] = tuple(s['value'] for s in overrideState.values())
#         return state

#     def _interpretValue(self, v):
#         return self.mkPen(v)

#     def setValue(self, value, blockSignal=None):
#         if not fn.eq(value, self.pen):
#             value = self.mkPen(value)
#             PenSelectorDialog.updateParamFromPen(self.penOptsParam, value)
#         return super().setValue(self.pen, blockSignal)

#     def applyOptsToPen(self, **opts):
#         # Transform opts into a value for the current pen
#         paramNames = set(opts).intersection(self.penOptsParam.names)
#         # Value should be overridden by opts
#         with self.treeChangeBlocker():
#             if 'value' in opts:
#                 pen = self.mkPen(opts.pop('value'))
#                 if not fn.eq(pen, self.pen):
#                     PenSelectorDialog.updateParamFromPen(self.penOptsParam, pen)
#             penOpts = {}
#             for kk in paramNames:
#                 penOpts[kk] = opts[kk]
#                 self.penOptsParam[kk] = opts[kk]
#         return penOpts

#     def setOpts(self, **opts):
#         # Transform opts into a value
#         penOpts = self.applyOptsToPen(**opts)
#         if penOpts:
#             self.setValue(self.pen)
#         return super().setOpts(**opts)

#     def mkPen(self, *args, **kwargs):
#         """Thin wrapper around fn.mkPen which accepts the serialized state from saveState"""
#         if len(args) == 1 and isinstance(args[0], tuple) and len(args[0]) == len(self.penOptsParam.childs):
#             opts = dict(zip(self.penOptsParam.names, args[0]))
#             self.applyOptsToPen(**opts)
#             args = (self.pen,)
#             kwargs = {}
#         return fn.mkPen(*args, **kwargs)

# registerParameterType('pen', PenParameter, override=True)
# registerParameterType('progress', ProgressBarParameter, override=True)
# registerParameterType('file', FileParameter, override=True)
registerParameterType('slider', SliderParameter, override=True)
# registerParameterType('calendar', CalendarParameter, override=True)
# registerParameterType('font', FontParameter, override=True)
