from ..Qt import QtCore, QtGui, QtWidgets
from ..parametertree import Parameter, ParameterTree
from ..functions import mkPen

import re
from contextlib import ExitStack

class PenPreviewArea(QtWidgets.QLabel):
    def __init__(self, pen):
        super().__init__()
        self.penLocs = []
        self.lastPos = None
        self.pen = pen

    def mousePressEvent(self, ev):
        self.penLocs.clear()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        ret =super().mouseMoveEvent(ev)
        if not (ev.buttons() & QtCore.Qt.MouseButton.LeftButton):
            return ret
        pos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        if pos != self.lastPos:
            self.penLocs.append(pos)
            self.lastPos = QtCore.QPointF(pos)
            self.update()
        return ret

    def paintEvent(self, *args):
        painter = QtGui.QPainter(self)
        # draw a squigly line to show what the pen looks like.
        if len(self.penLocs) < 1:
            path = self.getDefaultPath()
        else:
            path = QtGui.QPainterPath()
            path.moveTo(self.penLocs[0])
            for pos in self.penLocs[1:]:
                path.lineTo(pos)

        painter.setPen(self.pen)
        painter.drawPath(path)
        painter.end()

    def getDefaultPath(self):
        w, h = self.width(), self.height()
        path = QtGui.QPainterPath()
        path.moveTo(w * .2, h * .2)
        path.lineTo(w * .8, h * .2)
        path.lineTo(w * .2, h * .5)
        path.cubicTo(w * .1, h * 1, w * .5, h * .25, w * .8, h * .8)
        return path

class PenSelectorDialog(QtWidgets.QDialog):
    def __init__(self, initialPen='k'):
        super().__init__()
        self.pen = mkPen(initialPen)
        self.param = self.mkParam(self.pen)
        self.tree = ParameterTree(showHeader=False)
        self.tree.setParameters(self.param, showTop=False)
        self.setupUi()
        self.setModal(True)

    @staticmethod
    def mkParam(boundPen=None):
        # Import here to avoid cyclic dependency
        from ..parametertree.parameterTypes import QtEnumParameter
        cs = QtCore.Qt.PenCapStyle
        js = QtCore.Qt.PenJoinStyle
        ps = QtCore.Qt.PenStyle
        param = Parameter.create(name='Params', type='group', children=[
            dict(name='color', type='color', value='k'),
            dict(name='width', value=1, type='int', limits=[0, None]),
            QtEnumParameter(ps, name='style', value='SolidLine'),
            QtEnumParameter(cs, name='capStyle'),
            QtEnumParameter(js, name='joinStyle'),
            dict(name='cosmetic', type='bool', value=True)
        ])

        for p in param:
            name = p.name()
            replace = r'\1 \2'
            name = re.sub(r'(\w)([A-Z])', replace, name)
            name = name.title().strip()
            p.setOpts(title=name)
            
        def setterWrapper(setter):
            """Ignores the 'param' argument of sigValueChanged"""
            def newSetter(_, value):
                return setter(value)
            return newSetter

        if boundPen is not None:
            PenSelectorDialog.updateParamFromPen(param, boundPen)
            for p in param:
                setter, setName = PenSelectorDialog._setterForParam(p.name(), boundPen, returnName=True)
                # Instead, set the parameter which will signal the old setter
                setattr(boundPen, setName, p.setValue)
                p.sigValueChanged.connect(setterWrapper(setter))
                # Populate initial value
        return param

    @staticmethod
    def updatePenFromParam(penOptsParam, pen=None):
        if pen is None:
            pen = mkPen()
        for param in penOptsParam:
            setter = PenSelectorDialog._setterForParam(param.name(), pen)
            setter(param.value())
        return pen

    def updatePenFromOpts(self, penOpts, pen=None):
        if pen is None:
            pen = mkPen()
        useKeys = set(penOpts).intersection(self.param.names)
        for kk in useKeys:
            setter = self._setterForParam(kk, pen)
            setter(penOpts[kk])

    @staticmethod
    def _setterForParam(paramName, obj, returnName=False):
        formatted = paramName[0].upper() + paramName[1:]
        setter = getattr(obj, f'set{formatted}')
        if returnName:
            return setter, formatted
        return setter

    @staticmethod
    def updateParamFromPen(param, pen):
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

    def setupUi(self):
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tree)

        self.buttonBoxAcceptCancel = QtWidgets.QDialogButtonBox(self)
        self.buttonBoxAcceptCancel.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttonBoxAcceptCancel.setStandardButtons(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.buttonBoxAcceptCancel.accepted.connect(self.accept)
        self.buttonBoxAcceptCancel.rejected.connect(self.reject)

        self.labelPenPreview = PenPreviewArea(self.pen)
        def maybeUpdatePreview(_, changes):
            if any('value' in c[1] for c in changes):
                self.labelPenPreview.update()
        self.param.sigTreeStateChanged.connect(maybeUpdatePreview)
        infoLbl = QtWidgets.QLabel('Click and drag below to test the pen')
        infoLbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        policy = QtGui.QSizePolicy.Policy
        infoLbl.setSizePolicy(policy.Expanding, policy.Fixed)
        self.labelPenPreview.setMinimumSize(10,30)
        self.tree.setMinimumSize(240, 135)
        self.tree.setMaximumHeight(135)

        layout.addWidget(infoLbl)
        layout.addWidget(self.labelPenPreview)
        layout.addWidget(self.buttonBoxAcceptCancel)

        self.setLayout(layout)
        self.resize(240, 300)
