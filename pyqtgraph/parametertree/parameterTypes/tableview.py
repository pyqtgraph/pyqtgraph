import importlib
import json
from qtpy import QtWidgets, QtCore
from collections import OrderedDict
from pyqtgraph.parametertree.Parameter import ParameterItem
from .basetypes import WidgetParameterItem
from .basetypes import Parameter
from ..xml_parameter_factory import XMLParameterFactory, XMLParameter

from pyqtgraph.widgets import ColorButton



class TableViewCustom(QtWidgets.QTableView):
    """
        ============== ===========================
        *Attributes**    **Type**
        *valuechanged*   instance of pyqt Signal
        *QtWidgets*      instance of QTableWidget
        ============== ===========================
    """

    valueChanged = QtCore.Signal(list)
    add_data_signal = QtCore.Signal(int)
    remove_row_signal = QtCore.Signal(int)
    load_data_signal = QtCore.Signal()
    save_data_signal = QtCore.Signal()

    def __init__(self, menu=False):
        super().__init__()
        self.setmenu(menu)

    def setmenu(self, status):
        if status:
            self.menu = QtWidgets.QMenu()
            self.menu.addAction('Add new', self.add)
            self.menu.addAction('Remove selected row', self.remove)
            self.menu.addAction('Clear all', self.clear)
            self.menu.addSeparator()
            self.menu.addAction('Load as txt', lambda: self.load_data_signal.emit())
            self.menu.addAction('Save as txt', lambda: self.save_data_signal.emit())
        else:
            self.menu = None

    def clear(self):
        self.model().clear()

    def add(self):
        self.add_data_signal.emit(self.currentIndex().row())

    def remove(self):
        self.remove_row_signal.emit(self.currentIndex().row())

    def data_has_changed(self, topleft, bottomright, roles):
        self.valueChanged.emit([topleft, bottomright, roles])

    def get_table_value(self):
        """

        """
        return self.model()

    def set_table_value(self, data_model):
        """

        """
        try:
            self.setModel(data_model)
            self.model().dataChanged.connect(self.data_has_changed)
        except Exception as e:
            pass

    def contextMenuEvent(self, event):
        if self.menu is not None:
            self.menu.exec(event.globalPos())


class TableViewParameterItem(WidgetParameterItem):
    def makeWidget(self):
        """
            Make and initialize an instance of Table_custom.

            Returns
            -------
            table : instance of Table_custom.
                The initialized table.

            See Also
            --------
            Table_custom
        """
        self.asSubItem = True
        self.hideWidget = False
        menu = False
        opts = self.param.opts
        if 'menu' in opts:
            menu = opts['menu']
        w = TableViewCustom(menu=menu)

        if 'tip' in opts:
            w.setToolTip(opts['tip'])
        w.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        #w.setMaximumHeight(200)
        # self.table.setReadOnly(self.param.opts.get('readonly', False))
        w.value = w.get_table_value
        w.setValue = w.set_table_value
        w.sigChanged = w.valueChanged
        return w

    def optsChanged(self, param, opts):
        """
            | Called when any options are changed that are not name, value, default, or limits.
            |
            | If widget is a SpinBox, pass options straight through.
            | So that only the display label is shown when visible option is toggled.

            =============== ================================== ==============================
            **Parameters**    **Type**                           **Description**
            *param*           instance of pyqtgraph parameter    the parameter to check
            *opts*            string list                        the associated options list
            =============== ================================== ==============================

            See Also
            --------
            optsChanged
        """
        # print "opts changed:", opts
        ParameterItem.optsChanged(self, param, opts)

        if 'delegate' in opts:
            styledItemDelegate = QtWidgets.QStyledItemDelegate()
            styledItemDelegate.setItemEditorFactory(opts['delegate']())
            self.widget.setItemDelegate(styledItemDelegate)

        if 'menu' in opts:
            self.widget.setmenu(opts['menu'])


class TableViewParameter(Parameter, XMLParameter):
    """
        =============== =================================
        **Attributes**    **Type**
        *itemClass*       instance of TableParameterItem
        *Parameter*       instance of pyqtgraph parameter
        =============== =================================
    """
    itemClass = TableViewParameterItem

    def setValue(self, value):
        self.opts['value'] = value
        self.sigValueChanged.emit(self, value)

    @staticmethod
    def set_specific_options(el):
        param_dict = {}
        value = el.get('value','')
        param_dict['show_pb'] = True if el.get('show_pb', '0') == '1' else False
        
        data_dict = json.loads(value)
        mod = importlib.import_module(data_dict['module'])
        _cls = getattr(mod, data_dict['classname'])
        param_dict['value'] = _cls(data_dict['data'], header=data_dict['header'])

        return param_dict
        
    @staticmethod
    def get_specific_options(param):
        param_value = param.opts.get('value', None)
        try:
            data = dict(classname=param_value.__class__.__name__,
                        module=param_value.__class__.__module__,
                        data=param_value.get_data_all(),
                        header=param_value.header)
            value = json.dumps(data)
        except Exception:
            value = ''

        opts = {
            "value": value,
        }
    
        return opts