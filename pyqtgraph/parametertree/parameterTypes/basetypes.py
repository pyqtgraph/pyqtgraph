import builtins

from PyQt6.uic.Compiler.qtproxies import QtGui

from ... import functions as fn
from ... import icons
from ...Qt import QtCore, QtWidgets
from ...Qt.QtGui import QColor
from ..Parameter import Parameter
from ..ParameterItem import ParameterItem

from xml.etree.ElementTree import Element

class WidgetParameterItem(ParameterItem):
    """
    ParameterTree item with:

      * label in second column for displaying value
      * simple widget for editing value (displayed instead of label when item is selected)
      * button that resets value to default

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

        This function must be overridden by a subclass.
        """
        raise NotImplementedError

    def widgetEventFilter(self, obj, ev):
        ## filter widget's events
        ## catch TAB to change focus
        ## catch focusOut to hide editor
        if ev.type() == ev.Type.KeyPress:
            if ev.key() == QtCore.Qt.Key.Key_Tab:
                self.focusNext(forward=True)
                return True  ## don't let anyone else see this event
            elif ev.key() == QtCore.Qt.Key.Key_Backtab:
                self.focusNext(forward=False)
                return True  ## don't let anyone else see this event

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
        self.defaultBtn.setEnabled(
            self.param.valueModifiedSinceResetToDefault()
            and self.param.opts['enabled']
            and self.param.writable())

        self.defaultBtn.setVisible(self.param.hasDefault() and not self.param.readonly())

    def updateDisplayLabel(self, value=None):
        """Update the display label to reflect the value of the parameter."""
        if value is None:
            value = self.param.value()
        self.displayLabel.setText(str(value))

    def widgetValueChanged(self):
        ## called when the widget's value has been changed by the user
        val = self.widget.value()
        self.param.setValue(val)

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

    def hideEditor(self):
        self.widget.hide()
        self.displayLabel.show()

    def limitsChanged(self, param, limits):
        """Called when the parameter's limits have changed"""
        ParameterItem.limitsChanged(self, param, limits)

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
        self.updateDefaultBtn()

    def optsChanged(self, param, opts):
        """Called when any options are changed that are not
        name, value, default, or limits"""
        ParameterItem.optsChanged(self, param, opts)

        if 'enabled' in opts:
            self.updateDefaultBtn()
            self.widget.setEnabled(opts['enabled'])

        if 'readonly' in opts:
            self.updateDefaultBtn()

            if opts['readonly']:
                self.displayLabel.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            else:
                self.displayLabel.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse)

            if hasattr(self.widget, 'setReadOnly'):
                self.widget.setReadOnly(opts['readonly'])
            else:
                self.widget.setEnabled(self.param.opts['enabled'] and not opts['readonly'])

        if 'tip' in opts:
            self.widget.setToolTip(opts['tip'])


class EventProxy(QtCore.QObject):
    def __init__(self, qobj, callback):
        QtCore.QObject.__init__(self)
        self.callback = callback
        qobj.installEventFilter(self)

    def eventFilter(self, obj, ev):
        return self.callback(obj, ev)


def el_value_to_dict(el: Element, attr = 'value') -> dict:
    param_dict = {}
    if attr in el.attrib:
        value = el.get(attr)
        if el.get('type') == 'bool':
            param_dict[attr] = True if value == '1' else False
        elif el.get('type') == 'int':
            param_dict[attr] = int(value)
        elif el.get('type') == 'float':
            param_dict[attr] = float(value)
        elif el.get('type') == 'str':
            param_dict[attr] = value
        elif el.get('type') == 'color':
            c = QColor()
            c.setRgba(int(value))
            param_dict[attr] = c
        else:
            raise TypeError(f'No interpreter found for type {el.get("type")}')
    return param_dict


class SimpleParameter(Parameter):
    """
    Parameter representing a single value.

    This parameter is backed by :class:`~pyqtgraph.parametertree.parameterTypes.basetypes.WidgetParameterItem`
     to represent the following parameter names through various subclasses:

      - 'int'
      - 'float'
      - 'bool'
      - 'str'
      - 'color'
      - 'colormap'
    """

    @property
    def itemClass(self):
        from .bool import BoolParameterItem
        from .numeric import NumericParameterItem
        from .str import StrParameterItem
        return {
            'bool': BoolParameterItem,
            'int': NumericParameterItem,
            'float': NumericParameterItem,
            'str': StrParameterItem,
        }[self.opts['type']]


    @staticmethod
    def specific_options_from_xml(el):
        """
        Extract and convert a typed value from an XML element.

        This function retrieves the `value` attribute from the given XML element
        and converts it into a Python value based on the type specified in the `type` attribute.

        Args:
            el (xml.etree.ElementTree.Element): The XML element containing `type` and `value` attributes.

        Returns:
            dict: A dictionary containing the key `'value'` with the corresponding typed value.

        Raises:
            TypeError: If the type specified in the `type` attribute is unsupported.
        """
        param_dict = {}
        param_dict.update(el_value_to_dict(el, 'value'))
        param_dict.update(el_value_to_dict(el, 'default'))

        key = "limits"
        if key in el.attrib:
            value = eval(el.get(key))
            param_dict.update({key: value})

        return param_dict

    def value_to_dict(self, value) -> dict:
        opts = {}
        if self.opts['type'] == 'bool':
            opts['value'] = '1' if value else '0'
        elif self.opts['type'] == 'int':
            opts['value'] = str(value)
        elif self.opts['type'] == 'float':
            opts['value'] = str(value)
        elif self.opts['type'] == 'str':
            opts['value'] = value
        elif self.opts['type'] == 'color':
            opts['value'] = str(value.rgba())
        else:
            raise TypeError(f'No serializer found for type {self.opts["type"]}')
        return opts

    def specific_options_from_parameter(self):
        """
        Convert a parameter's value into a format compatible with XML representation.

        This function extracts the value from a `Parameter` object and formats it 
        according to the type specified in `param.opts['type']`. The result is suitable 
        for insertion into an XML element's `value` attribute.

        Args:
            param (pyqtgraph.parametertree.Parameter): The `Parameter` object containing 
                the value and type to interpret.

        Returns:
            dict: A dictionary containing the key `'value'` with the formatted value as a string.

        Raises:
            TypeError: If the type specified in `param.opts['type']` is unsupported.
        """
        param_value = self.opts.get('value', None)
        opts = self.value_to_dict(param_value)
        if 'default' in self.opts and self.opts['default'] is not None:
            opts.update(self.value_to_dict(self.opts['default']))

        key = "limits"
        if key in self.opts:
            opts[key] = str(self.opts[key])


        return opts
    
    def _interpretValue(self, v):
        typ = self.opts['type']

        def _missing_interp(v):
            # Assume raw interpretation
            return v
            # Or:
            # raise TypeError(f'No interpreter found for type {typ}')

        interpreter = getattr(builtins, typ, _missing_interp)
        return interpreter(v)
    
    


class GroupParameterItem(ParameterItem):
    """
    Group parameters are used mainly as a generic parent item that holds (and groups!) a set
    of child parameters. It also provides a simple mechanism for displaying a button or combo
    that can be used to add new parameters to the group.
    """

    def __init__(self, param, depth):
        ParameterItem.__init__(self, param, depth)
        self._initialFontPointSize = self.font(0).pointSize()
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
            l.setContentsMargins(0, 0, 0, 0)
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

    def pointSize(self):
        return self._initialFontPointSize

    def updateDepth(self, depth):
        """
        Change set the item font to bold and increase the font size on outermost groups.
        """
        for c in [0, 1]:
            font = self.font(c)
            font.setBold(True)
            if depth == 0:
                font.setPointSize(self.pointSize() + 1)
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
            ParameterItem.insertChild(self, self.childCount() - 1, child)
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

    @staticmethod
    def specific_options_from_xml(el):
        """
        Extract and convert a typed value from an XML element.

        This function retrieves the `value` attribute from the given XML element
        and converts it into a Python value based on the type specified in the `type` attribute.

        Args:
            el (xml.etree.ElementTree.Element): The XML element containing `type` and `value` attributes.

        Returns:
            dict: A dictionary containing the key `'value'` with the corresponding typed value.

        Raises:
            TypeError: If the type specified in the `type` attribute is unsupported.
        """
        return {}

    def specific_options_from_parameter(self):
        """
        Convert a parameter's value into a format compatible with XML representation.

        This function extracts the value from a `Parameter` object and formats it
        according to the type specified in `param.opts['type']`. The result is suitable
        for insertion into an XML element's `value` attribute.

        Args:
            param (pyqtgraph.parametertree.Parameter): The `Parameter` object containing
                the value and type to interpret.

        Returns:
            dict: A dictionary containing the key `'value'` with the formatted value as a string.

        Raises:
            TypeError: If the type specified in `param.opts['type']` is unsupported.
        """
        return {}



class Emitter(QtCore.QObject):
    """
    WidgetParameterItem is not a QObject, so create a QObject wrapper that items can use for emitting
    """
    sigChanging = QtCore.Signal(object, object)
    sigChanged = QtCore.Signal(object, object)
