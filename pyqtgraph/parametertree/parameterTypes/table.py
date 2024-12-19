from qtpy import QtWidgets, QtCore
from collections import OrderedDict
from .basetypes import WidgetParameterItem
from ..xml_parameter_factory import XMLParameterFactory, XMLParameter

from .basetypes import Parameter


class TableWidget(QtWidgets.QTableWidget):
    """
        ============== ===========================
        *Attributes**    **Type**
        *valuechanged*   instance of pyqt Signal
        *QtWidgets*      instance of QTableWidget
        ============== ===========================
    """

    valuechanged = QtCore.Signal(OrderedDict)

    def __init__(self):
        super().__init__()

    def get_table_value(self):
        """
            Get the contents of the self coursed table.

            Returns
            -------
            data : ordered dictionnary
                The getted values dictionnary.
        """
        data = OrderedDict([])
        for ind in range(self.rowCount()):
            item0 = self.item(ind, 0)
            item1 = self.item(ind, 1)
            if item0 is not None and item1 is not None:
                try:
                    data[item0.text()] = float(item1.text())
                except Exception:
                    data[item0.text()] = item1.text()
        return data

    def set_table_value(self, data_dict):
        """
            Set the data values dictionnary to the custom table.

            =============== ====================== ================================================
            **Parameters**    **Type**               **Description**
            *data_dict*       ordered dictionnary    the contents to be stored in the custom table
            =============== ====================== ================================================
        """
        try:
            self.setRowCount(len(data_dict))
            self.setColumnCount(2)
            for ind, (key, value) in enumerate(data_dict.items()):
                item0 = QtWidgets.QTableWidgetItem(key)
                item0.setFlags(item0.flags() ^ QtCore.Qt.ItemIsEditable)
                if isinstance(value, float):
                    item1 = QtWidgets.QTableWidgetItem('{:.3e}'.format(value))
                else:
                    item1 = QtWidgets.QTableWidgetItem(str(value))
                item1.setFlags(item1.flags() ^ QtCore.Qt.ItemIsEditable)
                self.setItem(ind, 0, item0)
                self.setItem(ind, 1, item1)
            # self.valuechanged.emit(data_dict)

        except Exception as e:
            pass


class TableParameterItem(WidgetParameterItem):

    # def treeWidgetChanged(self):
    #     """
    #         Check for changement in the Widget tree.
    #     """
    #     # # TODO: fix so that superclass method can be called
    #     # # (WidgetParameter should just natively support this style)
    #     # WidgetParameterItem.treeWidgetChanged(self)
    #     self.treeWidget().setFirstItemColumnSpanned(self.subItem, True)
    #     self.treeWidget().setItemWidget(self.subItem, 0, self.widget)
    #
    #     # for now, these are copied from ParameterItem.treeWidgetChanged
    #     self.setHidden(not self.param.opts.get('visible', True))
    #     self.setExpanded(self.param.opts.get('expanded', True))

    def makeWidget(self):
        """
            Make and initialize an instance of TableWidget.

            Returns
            -------
            table : instance of TableWidget.
                The initialized table.

            See Also
            --------
            TableWidget
        """
        self.asSubItem = True
        self.hideWidget = False
        opts = self.param.opts
        w = TableWidget()
        if 'tip' in opts:
            w.setToolTip(opts['tip'])
        w.setColumnCount(2)
        if 'header' in opts:
            w.setHorizontalHeaderLabels(self.param.opts['header'])
        if 'height' not in opts:
            opts['height'] = 200
        w.setMaximumHeight(opts['height'])
        w.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        # self.table.setReadOnly(self.param.opts.get('readonly', False))
        w.value = w.get_table_value
        w.setValue = w.set_table_value
        w.sigChanged = w.itemChanged
        return w


class TableParameter(Parameter, XMLParameter):
    """
        =============== =================================
        **Attributes**    **Type**
        *itemClass*       instance of TableParameterItem
        *Parameter*       instance of pyqtgraph parameter
        =============== =================================
    """
    itemClass = TableParameterItem
    """Editable string; displayed as large text box in the tree."""

    # def __init(self):
    #     super(TableParameter,self).__init__()

    def setValue(self, value):
        self.opts['value'] = value
        self.sigValueChanged.emit(self, value)

    @staticmethod
    def set_specific_options(el):
        param_dict = {}
        value = el.get('value','')
        param_dict['value'] = eval(value)
        return param_dict
    
    @staticmethod
    def get_specific_options(param):
        opts = {}
        value = param.opts.get('value', [])
        if isinstance(value, list):
            opts['value'] = repr(value)
        return opts