from OpenGL.GL import *
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem

class GLTextItem(GLGraphicsItem):
    """Draws text in 3D."""

    def __init__(self, **kwds):
        """All keyword arguments are passed to setData()"""
        GLGraphicsItem.__init__(self)
        glopts = kwds.pop('glOptions', 'additive')
        self.setGLOptions(glopts)

        self.pos = np.array([0.0, 0.0, 0.0])
        self.color = QtCore.Qt.white
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
                raise Exception('Invalid keyword argument: %s (allowed arguments are %s)' % (k, str(args)))
        for arg in args:
            if arg in kwds:
                value = kwds[arg]
                if arg == 'pos':
                    if isinstance(value, np.ndarray):
                        if value.shape != (3,):
                            raise Exception('"pos.shape" must be (3,).')
                    elif isinstance(value, (tuple, list)):
                        if len(value) != 3:
                            raise Exception('"len(pos)" must be 3.')
                elif arg == 'color':
                    if isinstance(value, QtGui.QColor):
                        pass
                    elif isinstance(value, (tuple, list, np.ndarray)):
                        if isinstance(value, np.ndarray):
                            if len(value.shape) != 1:
                                raise Exception('"color.shape" must be (3,) or (4,).')
                        value_len = len(value)
                        if value_len == 3:
                            value = QtGui.QColor(value[0], value[1], value[2])
                        elif value_len == 4:
                            value = QtGui.QColor(value[0], value[1], value[2], value[3])
                        else:
                            raise Exception('"len(color)" must be 3 or 4.')
                elif arg == 'font':
                    if isinstance(value, QtGui.QFont) is False:
                        raise Exception('"font" must be QFont.')
                setattr(self, arg, value)
        self.update()
    
    def paint(self):
        if len(self.text) < 1:
            return
        self.setupGLState()

        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)

        text_pos = self.__project(self.pos, modelview, projection, viewport)
        text_pos[1] = viewport[3] - text_pos[1]
        
        painter = QtGui.QPainter(self.view())
        painter.setPen(self.color)
        painter.setFont(self.font)
        painter.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
        painter.drawText(text_pos[0], text_pos[1], self.text)
        painter.end()
    
    def __project(self, obj_pos, modelview, projection, viewport):
        obj_vec = np.append(np.array(obj_pos), [1.0])

        view_vec = np.matmul(modelview.T, obj_vec)
        proj_vec = np.matmul(projection.T, view_vec)

        if proj_vec[3] == 0.0:
            return
        
        proj_vec[0:3] /= proj_vec[3]

        return np.array([
            viewport[0] + (1.0 + proj_vec[0]) * viewport[2] / 2.0,
            viewport[1] + (1.0 + proj_vec[1]) * viewport[3] / 2.0,
            (1.0 + proj_vec[2]) / 2.0
        ])
    