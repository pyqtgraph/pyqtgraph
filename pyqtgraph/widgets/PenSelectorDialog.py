from ..Qt import QtCore, QtGui, QtWidgets
from ..parametertree import Parameter, ParameterTree
from ..functions import mkPen

import re

class PenSelectorDialog(QtWidgets.QDialog):
    def __init__(self, initialPen='k'):
        super().__init__()
        self.param = self.mkParam()
        self.tree = ParameterTree(showHeader=False)
        self.tree.setParameters(self.param, showTop=False)
        self.setupUi()
        self.setModal(True)

        self.pen = mkPen(initialPen)
        self.updatePen(**self.param)

    def mkParam(self):
        cs = QtCore.Qt.PenCapStyle
        js = QtCore.Qt.PenJoinStyle
        ps = QtCore.Qt.PenStyle
        param = Parameter.create(name='Params', type='group', children=[
            dict(name='penWidth', value=1.0, type='float'),
            dict(name='penColor', type='color', value='k'),
            dict(name='penStyle', type='list', limits={
                'Solid': ps.SolidLine,
                'Dashed': ps.DashLine,
                'Dash dot': ps.DashDotLine,
                'Dash dot dot': ps.DashDotDotLine
            }),
            dict(name='penCapStyle', type='list', limits={
                'Square cap': cs.SquareCap,
                'Flat cap': cs.FlatCap,
                'Round cap': cs.RoundCap,
            }),
            dict(name='penJoinStyle', type='list', limits={
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
            p.sigValueChanged.connect(self.updatePenFromParam)
        return param

    def updatePenFromParam(self, param, val):
        kwargs = dict(self.param)
        kwargs[param.name()] = val
        self.updatePen(**kwargs)

    def setupUi(self):
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tree)

        self.buttonBoxAcceptCancel = QtWidgets.QDialogButtonBox(self)
        self.buttonBoxAcceptCancel.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttonBoxAcceptCancel.setStandardButtons(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.buttonBoxAcceptCancel.accepted.connect(self.accept)
        self.buttonBoxAcceptCancel.rejected.connect(self.reject)

        self.labelPenPreview = QtWidgets.QLabel()
        self.labelPenPreview.setMinimumSize(10,30)
        self.tree.setMinimumSize(240, 115)
        self.tree.setMaximumHeight(115)

        layout.addWidget(self.labelPenPreview)
        layout.addWidget(self.buttonBoxAcceptCancel)

        self.setLayout(layout)
        self.resize(240, 300)

    def updatePen(self, penColor, penWidth, penStyle, penCapStyle, penJoinStyle):
        self.pen = QtGui.QPen(penColor, penWidth, penStyle, penCapStyle, penJoinStyle)

        displaySize = self.labelPenPreview.size()
        w, h = displaySize.width(), displaySize.height()
        palette = self.labelPenPreview.palette()
        labelBackgroundColor = palette.color(palette.ColorRole.Window)

        # draw a squigly line to show what the pen looks like.
        path = QtGui.QPainterPath()
        path.moveTo(w * .2, h * .2)
        path.lineTo(w * .8, h * .2)
        path.lineTo(w * .2, h * .5)
        path.cubicTo(w * .1, h * 1, w * .5, h * .25, w * .8, h * .8)

        p = QtGui.QPixmap(displaySize)
        p.fill(labelBackgroundColor)
        painter = QtGui.QPainter(p)
        painter.setPen(self.pen)
        painter.drawPath(path)
        painter.end()
        self.labelPenPreview.setPixmap(p)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.updatePen(**self.param)