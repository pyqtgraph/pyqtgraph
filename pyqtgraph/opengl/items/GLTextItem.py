from OpenGL.GL import *  # noqa
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
        ====================  ==================================================
        """
        args = ['pos', 'color', 'text', 'font']
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

        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)

        viewport = [0, 0, self.view().width(), self.view().height()]
        text_pos = self.__project(self.pos, modelview, projection, viewport)

        text_pos.setY(viewport[3] - text_pos.y())

        painter = QtGui.QPainter(self.view())
        painter.setPen(self.color)
        painter.setFont(self.font)
        painter.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing)
        painter.drawText(text_pos, self.text)
        painter.end()

    def __project(self, obj_pos, modelview, projection, viewport):
        obj_vec = np.append(np.array(obj_pos), [1.0])

        view_vec = np.matmul(modelview.T, obj_vec)
        proj_vec = np.matmul(projection.T, view_vec)

        if proj_vec[3] == 0.0:
            return QtCore.QPointF(0, 0)

        proj_vec[0:3] /= proj_vec[3]

        return QtCore.QPointF(
            viewport[0] + (1.0 + proj_vec[0]) * viewport[2] / 2,
            viewport[1] + (1.0 + proj_vec[1]) * viewport[3] / 2
        )
