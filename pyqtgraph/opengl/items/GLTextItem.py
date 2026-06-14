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

        self.items = []
        self.pos = [0.0, 0.0, 0.0]
        self.color = QtGui.QColor(QtCore.Qt.GlobalColor.white)
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
        text                  String to display.
        color                 QColor or array of ints [R,G,B] or [R,G,B,A]. (Default: Qt.white)
        font                  QFont (Default: QFont('Helvetica', 16))
        alignment             QtCore.Qt.AlignmentFlag (Default: QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignBottom)
        items                 Optional list of dicts. Each dict specifies parameters for a single textitem:
                              {'pos', 'text', 'color', 'font', 'alignment'}. This allows rendering multiple
                              text items. 'pos' and 'text' are required keys, the rest are optional.
        ====================  ==================================================
        """
        args = ['pos', 'text', 'color', 'font', 'alignment', 'items']
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
                elif arg == 'items':
                    if not isinstance(value, list):
                        raise ValueError('"items" must be list of dicts')
                setattr(self, arg, value)
        self.update()

    def paint(self):
        items = [dict(text=self.text, pos=self.pos)] if len(self.text) != 0 else self.items
        if len(items) == 0:
            return

        self.setupGLState()

        device = self.view()
        rect = QtCore.QRectF(0, 0, device.width(), device.height())
        project = self.compute_projection(rect)
        painter = QtGui.QPainter(device)
        painter.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing)

        for item in items:
            # text and pos must be specified
            text = item['text']
            pos = item['pos']

            if len(text) == 0:
                continue

            # alignment, color, font fallback to defaults
            alignment = item.get('alignment', self.alignment)
            color = item.get('color')
            color = fn.mkColor(color) if color is not None else self.color
            font = item.get('font', self.font)

            vec3 = QtGui.QVector3D(*pos)
            pos_2d = self.align_text(project.map(vec3).toPointF(), text, font, alignment)
            painter.setPen(color)
            painter.setFont(font)
            painter.drawText(pos_2d, text)

        painter.end()

    def compute_projection(self, rect : QtCore.QRectF):
        # note that QRectF.bottom() != QRect.bottom()
        ndc_to_viewport = QtGui.QMatrix4x4()
        ndc_to_viewport.viewport(rect.left(), rect.bottom(), rect.width(), -rect.height())
        return ndc_to_viewport * self.mvpMatrix()

    def align_text(self, pos, text, font, alignment):
        """
        Aligns the text at the given position according to the given alignment.
        """
        font_metrics = QtGui.QFontMetrics(font)
        rect = font_metrics.tightBoundingRect(text)
        width = rect.width()
        height = rect.height()
        dx = dy = 0.0
        if alignment & QtCore.Qt.AlignmentFlag.AlignRight:
            dx = width
        if alignment & QtCore.Qt.AlignmentFlag.AlignHCenter:
            dx = width / 2.0
        if alignment & QtCore.Qt.AlignmentFlag.AlignTop:
            dy = height
        if alignment & QtCore.Qt.AlignmentFlag.AlignVCenter:
            dy = height / 2.0
        return QtCore.QPointF(pos.x() - dx, pos.y() + dy)
