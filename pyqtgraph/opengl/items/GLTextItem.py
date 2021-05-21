from typing import List, Tuple, Union
from OpenGL.GL import *
from OpenGL.GLUT import *
import numpy as np
import pyqtgraph.functions as fn
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem

class GLTextItem(GLGraphicsItem):
    """Draws text in 3D."""

    def __init__(self, **kwds):
        """All keyword arguments are passed to setData()"""
        GLGraphicsItem.__init__(self)
        glopts = kwds.pop('glOptions', 'additive')
        self.setGLOptions(glopts)

        self.pos = np.array([0.0, 0.0, 0.0])
        self.color = (1.0, 1.0, 1.0, 1.0)
        self.text = ''
        self.font = GLUT_BITMAP_HELVETICA_18

        self.setData(**kwds)
        glutInit()
    
    def setData(self, **kwds):
        """
        Update the data displayed by this item. All arguments are optional;
        for example it is allowed to update text while leaving colors unchanged, etc.

        ====================  ==================================================
        **Arguments:**
        ------------------------------------------------------------------------
        pos                   (3,) array of floats specifying text location.
        color                 (4,) array of floats (0.0-1.0) or
                              tuple of floats specifying
                              a single color for the entire item.
        text                  String to display.
        font                  GLUT font. (Default: GLUT_BITMAP_HELVETICA_18)
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
                    value = fn.glColor(value)
                elif arg == 'font':
                    if value not in [
                        GLUT_BITMAP_8_BY_13,
                        GLUT_BITMAP_9_BY_15,
                        GLUT_BITMAP_TIMES_ROMAN_10,
                        GLUT_BITMAP_TIMES_ROMAN_24,
                        GLUT_BITMAP_HELVETICA_10,
                        GLUT_BITMAP_HELVETICA_12,
                        GLUT_BITMAP_HELVETICA_18
                    ]:
                        raise Exception('"font" must be ' \
                            '"GLUT_BITMAP_8_BY_13", ' \
                            '"GLUT_BITMAP_9_BY_15", ' \
                            '"GLUT_BITMAP_TIMES_ROMAN_10", ' \
                            '"GLUT_BITMAP_TIMES_ROMAN_24", ' \
                            '"GLUT_BITMAP_HELVETICA_10", ' \
                            '"GLUT_BITMAP_HELVETICA_12", or ' \
                            '"GLUT_BITMAP_HELVETICA_18".')
                setattr(self, arg, value)
        self.update()
    
    def paint(self):
        if len(self.text) < 1:
            return
        self.setupGLState()

        glColor4d(*self.color)
        glRasterPos3d(float(self.pos[0]), float(self.pos[1]), float(self.pos[2]))
        for char in self.text:
            glutBitmapCharacter(self.font, ord(char))
