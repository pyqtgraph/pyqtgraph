from ..Qt import QtCore, QtGui, QtWidgets
from ..parametertree import Parameter, ParameterTree
from ..functions import mkPen

import re


class PenPreviewArea(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.penLocs = []
        self.lastPos = None
        self.pen = None

    def mousePressEvent(self, ev):
        self.penLocs.clear()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        ret =super().mouseMoveEvent(ev)
        if not (ev.buttons() & QtCore.Qt.MouseButton.LeftButton):
            return ret
        pos = ev.pos()
        if pos != self.lastPos:
            self.penLocs.append(pos)
            self.lastPos = QtCore.QPointF(pos)
            self.update()
        return ret

    def setPen(self, pen):
        self.pen = pen
        self.update()

    def paintEvent(self, *args):
        displaySize = self.size()
        w, h = displaySize.width(), displaySize.height()
        palette = self.palette()
        labelBackgroundColor = palette.color(palette.ColorRole.Window)
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
        self.param = self.mkParam()
        for p in self.param:
            p.sigValueChanged.connect(self.applyPenOpts)
        self.tree = ParameterTree(showHeader=False)
        self.tree.setParameters(self.param, showTop=False)
        self.setupUi()
        self.setModal(True)

        self.pen = mkPen(initialPen)
        self.updatePen(**self.param)

    @staticmethod
    def mkParam():
        cs = QtCore.Qt.PenCapStyle
        js = QtCore.Qt.PenJoinStyle
        ps = QtCore.Qt.PenStyle
        param = Parameter.create(name='Params', type='group', children=[
            dict(name='width', value=1.0, type='float'),
            dict(name='color', type='color', value='k'),
            dict(name='style', type='list', limits={
                'Solid': ps.SolidLine,
                'Dashed': ps.DashLine,
                'Dash dot': ps.DashDotLine,
                'Dash dot dot': ps.DashDotDotLine
            }),
            dict(name='capStyle', type='list', limits={
                'Square cap': cs.SquareCap,
                'Flat cap': cs.FlatCap,
                'Round cap': cs.RoundCap,
            }),
            dict(name='joinStyle', type='list', limits={
                'Bevel join': js.BevelJoin,
                'Miter join': js.MiterJoin,
                'Round join': js.RoundJoin
            })

        ])

        for p in param:
            name = p.name()
            replace = r'\1 \2'
            name = re.sub(r'(\w)([A-Z])', replace, name)
            name = name.title().strip()
            p.setOpts(title=name)
        return param

    @staticmethod
    def updatePenFromParam(pen, penOptsParam):
        for param in penOptsParam:
            formatted = param.name()
            formatted = formatted[0].upper() + formatted[1:]
            setter = getattr(pen, f'set{formatted}')
            setter(param.value())

    def applyPenOpts(self):
        self.updatePen(**self.param)

    @staticmethod
    def updateParamFromPen(param, pen):
        """
        Applies settings from a pen to either a Parameter or dict. The Parameter or dict must already
        be populated with the relevant keys that can be found in `PenSelectorDialog.mkParam`.
        """
        names = param if isinstance(param, dict) else param.names
        for opt in names:
            param[opt] = getattr(pen, opt)()

    def setupUi(self):
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tree)

        self.buttonBoxAcceptCancel = QtWidgets.QDialogButtonBox(self)
        self.buttonBoxAcceptCancel.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttonBoxAcceptCancel.setStandardButtons(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.buttonBoxAcceptCancel.accepted.connect(self.accept)
        self.buttonBoxAcceptCancel.rejected.connect(self.reject)

        self.labelPenPreview = PenPreviewArea()
        infoLbl = QtWidgets.QLabel('Click and drag below to test the pen')
        infoLbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        policy = QtGui.QSizePolicy.Policy
        infoLbl.setSizePolicy(policy.Expanding, policy.Fixed)
        self.labelPenPreview.setMinimumSize(10,30)
        self.tree.setMinimumSize(240, 115)
        self.tree.setMaximumHeight(115)

        layout.addWidget(infoLbl)
        layout.addWidget(self.labelPenPreview)
        layout.addWidget(self.buttonBoxAcceptCancel)

        self.setLayout(layout)
        self.resize(240, 300)

    def updatePen(self, color, width, style, capStyle, joinStyle):
        self.pen = QtGui.QPen(color, width, style, capStyle, joinStyle)
        self.labelPenPreview.setPen(self.pen)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.updatePen(**self.param)