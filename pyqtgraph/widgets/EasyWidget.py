from ..parametertree import Parameter
from ..parametertree.parameterTypes import WidgetParameterItem
from ..Qt import QtCore, QtWidgets
from .. import functions as fn

def hookupParamWidget(param: Parameter, widget):
    """
    Parameter widgets created outside param trees need to have their sigChanging, sigChanged, etc. signals hooked up to
    the parameter itself manually. The relevant code below was extracted from WidgetParameterItem
    """
    def widgetValueChanged():
        val = widget.value()
        return param.setValue(val)
    def paramValueChanged(param, val, force=False):
        if force or not fn.eq(val, widget.value()):
            try:
                widget.sigChanged.disconnect(widgetValueChanged)
                param.sigValueChanged.disconnect(paramValueChanged)
                widget.setValue(val)
                param.setValue(widget.value())
            finally:
                widget.sigChanged.connect(widgetValueChanged)
                param.sigValueChanged.connect(paramValueChanged)

    param.sigValueChanged.connect(paramValueChanged)
    if widget.sigChanged is not None:
        widget.sigChanged.connect(widgetValueChanged)

    if hasattr(widget, 'sigChanging'):
        widget.sigChanging.connect(lambda: param.sigValueChanging.emit(param, widget.value()))

    ## update value shown in widget.
    opts = param.opts
    if opts.get('value', None) is not None:
        paramValueChanged(param, opts['value'], force=True)
    else:
        ## no starting value was given; use whatever the widget has
        widgetValueChanged()

class EasyWidget:
    def __init__(
        self,
        children,
        layout: str=None,
        splitter=False,
        baseWidget=None
    ):
        if baseWidget is None:
            baseWidget = QtWidgets.QWidget()
        self._built = False
        self.children_ = children
        self.useSplitter = None
        self.widget_ = baseWidget
        self.layout_ = None

        self._resetOpts(splitter, layout)

    def _resetOpts(self, useSplitter, layout):
        if layout == 'V':
            orient = QtCore.Qt.Orientation.Vertical
            layout = QtWidgets.QVBoxLayout
        elif layout == 'H':
            orient = QtCore.Qt.Orientation.Horizontal
            layout = QtWidgets.QHBoxLayout
        else:
            orient = layout = None
        self.orient_ = orient

        if useSplitter == self.useSplitter and self.layout_:
            return
        # Had children in existing widget which will be discarded when changing self widget_ to splitter
        if self.widget_.children() and useSplitter:
            raise ValueError("Cannot change splitter status to *True* when widget already has children")
        self.useSplitter = useSplitter

        if useSplitter:
            self.layout_ = QtWidgets.QSplitter(orient)
            self.widget_ = self.layout_
        else:
            try:
                self.layout_ = layout()
                self.widget_.setLayout(self.layout_)
            except TypeError:
                # When layout is none
                self.layout_ = None

    def build(self):
        if self._built:
            return
        if self.layout_ is None:
            raise ValueError('Top-level orientation must be set to "V" or "H" before adding children')
        if self.orient_ == QtCore.Qt.Orientation.Horizontal:
            chSuggested = 'V'
        elif self.orient_ == QtCore.Qt.Orientation.Vertical:
            chSuggested = 'H'
        else:
            chSuggested = None

        for ii, child in enumerate(self.children_):
            morphChild = self.addChild(child, chSuggested)
            if morphChild is not child:
                self.children_[ii] = morphChild
        self._built = True

    def addChild(self, child, suggestedLayout:str=None):
        """Adds a child or list of children to self's layout. Child is either a widget, EasyWidget, or sequence of them"""
        if isinstance(child, QtWidgets.QWidget):
            self.layout_.addWidget(child)
        else:
            child = self.listChildrenWrapper(child, suggestedLayout)
            # At this point, child should be an EasyWidget
            child.build()
            self.layout_.addWidget(child.widget_)
        return child

    def insertChild(self, child, index: int):
        """
        Called internally or allows intermediate layout index insertion

        Parameters
        __________
        child: EasyWidget
             Child to insert
        """
        child.build()
        return self.layout_.insertWidget(index, child.widget_)

    def hide(self):
        self.widget_.hide()

    def show(self):
        self.widget_.show()

    def removeInnerMargins(self):
        for ch in self.children_:
            if isinstance(ch, EasyWidget):
                ch.removeInnerMargins()
                lay = ch.widget_.layout()
                # layout_ != widget_.layout() for splitter
                if lay:
                    lay.setContentsMargins(0, 0, 0, 0)
                    lay.setSpacing(0)

    @classmethod
    def listChildrenWrapper(cls, children, maybeNewLayout: str=None):
        """Converts EasyWidget with ambiguous layout or list of widgets into an EasyWidget with concrete layout"""
        if not isinstance(children, EasyWidget):
            children = cls(children)
        if children.layout_ is None and maybeNewLayout is not None:
            children._resetOpts(children.useSplitter, maybeNewLayout)
        return children

    @classmethod
    def buildMainWin(cls, children, win: QtWidgets.QMainWindow=None, layout='V', **kwargs):
        if win is None:
            win = QtWidgets.QMainWindow()
        if not isinstance(children, EasyWidget):
            children = cls(children, layout=layout, **kwargs)

        children.build()
        win.easyChild = children
        win.setCentralWidget(children.widget_)
        children.removeInnerMargins()
        return win

    @classmethod
    def buildWidget(cls, children, layout='V', **kwargs):
        builder = cls(children, layout=layout, **kwargs)
        builder.build()
        retWidget = builder.widget_
        retWidget.easyChild = builder
        builder.removeInnerMargins()
        return retWidget

    @classmethod
    def fromParameter(cls, param: Parameter=None, layout='H', **opts):
        """
        Creates a form-style EasyWidget (name + edit widget) from pyqtgraph parameter options or a parameter directly.

        Parameters
        ----------
        param: Parameter
            Parameter to use, if it already exists. Otherwise, one is created from `opts` and returned.
        layout: str
            EasyWidget layout type, 'H' or 'V'
        opts:
            If `param` is unspecified, a parameter is created from these options instead and returned

        Returns
        -------
        Just the EasyWidget if `param` is provided, otherwise (EasyWidget, Parameter) tuple
        """
        returnParam = False
        if param is None:
            param = Parameter.create(**opts)
            returnParam = True
        item = param.makeTreeItem(0)
        if not isinstance(item, WidgetParameterItem):
            raise ValueError(
                f'{cls} can only create parameter editors from registered pg widgets whose items subclass'
                f' "WidgetParameterItem" and are in pyqtgraph\'s *PARAM_TYPES*.\nThese requirements are not met for type'
                f' "{param.type()}"'
            )
        editWidget = item.makeWidget()
        hookupParamWidget(param, editWidget)
        lbl = QtWidgets.QLabel(opts['name'])
        obj = cls([lbl, editWidget], layout)
        obj.build()
        if returnParam:
            return obj, param
        return obj
