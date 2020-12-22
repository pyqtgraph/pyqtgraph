import os

from ..Qt import QtCore, QtGui
from ..python2_3 import asUnicode
from .Parameter import Parameter, registerParameterType
from .ParameterItem import ParameterItem
from ..widgets.SpinBox import SpinBox
from ..widgets.ColorButton import ColorButton
from ..colormap import ColorMap
from .. import pixmaps as pixmaps
from .. import functions as fn
from ..pgcollections import OrderedDict


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
            self.subItem.setFlags(QtCore.Qt.NoItemFlags)
            self.addChild(self.subItem)

        self.defaultBtn = QtGui.QPushButton()
        self.defaultBtn.setAutoDefault(False)
        self.defaultBtn.setFixedWidth(20)
        self.defaultBtn.setFixedHeight(20)
        modDir = os.path.dirname(__file__)
        self.defaultBtn.setIcon(QtGui.QIcon(pixmaps.getPixmap('default')))
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
                defs['format'] = '{value:d}'
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
        if ev.type() == ev.KeyPress:
            if ev.key() == QtCore.Qt.Key_Tab:
                self.focusNext(forward=True)
                return True ## don't let anyone else see this event
            elif ev.key() == QtCore.Qt.Key_Backtab:
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
        self.widget.setFocus(QtCore.Qt.OtherFocusReason)
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
                tree.setFirstItemColumnSpanned(self.subItem, True)
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
    itemClass = WidgetParameterItem
    
    def __init__(self, *args, **kargs):
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
                self.addWidget.setSizeAdjustPolicy(QtGui.QComboBox.AdjustToContents)
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
            self.addItem.setFlags(QtCore.Qt.ItemIsEnabled)
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
        tw.setFirstItemColumnSpanned(self, True)
        if self.addItem is not None:
            tw.setItemWidget(self.addItem, 0, self.addWidgetBox)
            tw.setFirstItemColumnSpanned(self.addItem, True)
        
    def addChild(self, child):  ## make sure added childs are actually inserted before add btn
        if self.addItem is not None:
            ParameterItem.insertChild(self, self.childCount()-1, child)
        else:
            ParameterItem.addChild(self, child)
            
    def optsChanged(self, param, opts):
        ParameterItem.optsChanged(self, param, opts)
        
        if 'addList' in opts:
            self.updateAddList()

        if 'enabled' in opts and hasattr(self, 'addWidget'):
            self.addWidget.setEnabled(opts['enabled'])

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
        
        tree.setFirstItemColumnSpanned(self, True)
        tree.setItemWidget(self, 0, self.layoutWidget)

    def titleChanged(self):
        self.button.setText(self.param.title())
        self.setSizeHint(0, self.button.sizeHint())

    def optsChanged(self, param, opts):
        ParameterItem.optsChanged(self, param, opts)

        if 'enabled' in opts:
            self.button.setEnabled(opts['enabled'])

    def buttonClicked(self):
        self.param.activate()
        
class ActionParameter(Parameter):
    """Used for displaying a button within the tree."""
    itemClass = ActionParameterItem
    sigActivated = QtCore.Signal(object)
    
    def activate(self):
        self.sigActivated.emit(self)
        self.emitStateChanged('activated', None)
        
registerParameterType('action', ActionParameter, override=True)


class TextParameterItem(WidgetParameterItem):
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
    """Editable string; displayed as large text box in the tree."""
    itemClass = TextParameterItem


registerParameterType('text', TextParameter, override=True)
