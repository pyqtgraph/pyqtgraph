import numpy as np

from ... import functions as fn
from ...Qt import QtCore, QtGui
from ..GLGraphicsItem import GLGraphicsItem

__all__ = ['GLTextItem']

class GLTextItem(GLGraphicsItem):
    """Draws text in 3D."""

    def __init__(self, parentItem=None, **kwds):
        """All keyword arguments are passed to setData()"""
        super().__init__(parentItem=parentItem)
        glopts = kwds.pop('glOptions', 'additive')
        self.setGLOptions(glopts)

        self.pos = np.array([0.0, 0.0, 0.0])
        self.color = QtCore.Qt.GlobalColor.white
        self.text = ''
        self.font = QtGui.QFont('Helvetica', 16)
        self.alignment = QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignBottom

        self.setData(**kwds)

    def setData(self, **kwds):
        """
        Update the data displayed by this item. All arguments are optional;
        for example it is allowed to update text while leaving colors unchanged, etc.

        ====================  ==================================================
        **Arguments:**
        ------------------------------------------------------------------------
        pos                   (3,) array of floats specifying text location.
        color                 QColor or array of ints [R,G,B] or [R,G,B,A]. (Default: Qt.white)
        text                  String to display.
        font                  QFont (Default: QFont('Helvetica', 16))
        alignment             QtCore.Qt.AlignmentFlag (Default: QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignBottom)
        ====================  ==================================================
        """
        args = ['pos', 'color', 'text', 'font', 'alignment']
        for k in kwds.keys():
            if k not in args:
                raise ValueError('Invalid keyword argument: %s (allowed arguments are %s)' % (k, str(args)))
        for arg in args:
            if arg in kwds:
                value = kwds[arg]
                if arg == 'pos':
                    if isinstance(value, np.ndarray):
                        if value.shape != (3,):
                            raise ValueError('"pos.shape" must be (3,).')
                    elif isinstance(value, (tuple, list)):
                        if len(value) != 3:
                            raise ValueError('"len(pos)" must be 3.')
                elif arg == 'color':
                    value = fn.mkColor(value)
                elif arg == 'font':
                    if isinstance(value, QtGui.QFont) is False:
                        raise TypeError('"font" must be QFont.')
                setattr(self, arg, value)
        self.update()

    def paint(self):
        if len(self.text) < 1:
            return
        self.setupGLState()

        project = self.compute_projection()
        vec3 = QtGui.QVector3D(*self.pos)
        text_pos = self.align_text(project.map(vec3).toPointF())

        painter = QtGui.QPainter(self.view())
        painter.setPen(self.color)
        painter.setFont(self.font)
        painter.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing)
        painter.drawText(text_pos, self.text)
        painter.end()

    def compute_projection(self):
        # note that QRectF.bottom() != QRect.bottom()
        rect = QtCore.QRectF(self.view().rect())
        ndc_to_viewport = QtGui.QMatrix4x4()
        ndc_to_viewport.viewport(rect.left(), rect.bottom(), rect.width(), -rect.height())
        return ndc_to_viewport * self.mvpMatrix()

    def align_text(self, pos):
        """
        Aligns the text at the given position according to the given alignment.
        """
        font_metrics = QtGui.QFontMetrics(self.font)
        rect = font_metrics.tightBoundingRect(self.text)
        width = rect.width()
        height = rect.height()
        dx = dy = 0.0
        if self.alignment & QtCore.Qt.AlignmentFlag.AlignRight:
            dx = width
        if self.alignment & QtCore.Qt.AlignmentFlag.AlignHCenter:
            dx = width / 2.0
        if self.alignment & QtCore.Qt.AlignmentFlag.AlignTop:
            dy = height
        if self.alignment & QtCore.Qt.AlignmentFlag.AlignVCenter:
            dy = height / 2.0
        pos.setX(pos.x() - dx)
        pos.setY(pos.y() + dy)
        return pos
