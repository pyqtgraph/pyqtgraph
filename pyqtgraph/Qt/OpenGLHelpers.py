import importlib
import warnings

import numpy as np

from . import QT_LIB, QtGui, QtWidgets
from . import OpenGLConstants as GLC

if QT_LIB in ["PyQt5", "PySide2"]:
    QtOpenGL = QtGui
    QtOpenGLWidgets = QtWidgets
else:
    QtOpenGL = importlib.import_module(f'{QT_LIB}.QtOpenGL')
    QtOpenGLWidgets = importlib.import_module(f"{QT_LIB}.QtOpenGLWidgets")

__all__ = ["getFunctions", "GraphicsViewGLWidget"]

def getFunctions(context):
    glfn = None
    format = context.format()

    if QT_LIB in ["PySide2", "PySide6"]:
        glfn = context.extraFunctions()

    elif QT_LIB in ["PyQt5", "PyQt6"]:
        # PyQt5 has context.versionFunctions().
        #    however, when there are multiple GraphicsItems, the following bug occurs:
        #    all except one of the C++ objects of the returned versionFunctions() get
        #    deleted. i.e. in PyQt5, we are not able to cache the return value.
        # Qt6 has QOpenGLVersionFunctionsFactory().
        #    however with OpenGL ES: "versionFunctions: Not supported on OpenGL ES."
        # To overcome the above listed issues, we load the modules directly.

        # PyQt{5,6} only provides 2.0, 2.1, 4.1_Core, ES2.
        # ES2 module is present only if PyQt was compiled for GLES.
        # On Debian packaged PyQt5, python3-pyqt5.qtopengl needs to be installed.
        if context.isOpenGLES():
            # PyQt could have been compiled against OpenGL Desktop
            # but an OpenGL ES context was requested, so we use "2_0"
            # as a fallback.
            suffixes = ["ES2", "2_0"]
        elif format.version() >= (4, 1):
            suffixes = ["4_1_Core"]
        else:
            suffixes = ["2_1"]
        for suffix in suffixes:
            glfnname = f"QOpenGLFunctions_{suffix}"
            try:
                modname = f"_{glfnname}" if QT_LIB == "PyQt5" else "QtOpenGL"
                QtOpenGLFunctions = importlib.import_module(f"{QT_LIB}.{modname}")
                glfn = getattr(QtOpenGLFunctions, glfnname)()
            except (ModuleNotFoundError, AttributeError):
                continue
            glfn.initializeOpenGLFunctions()
            break

    if glfn is None:
        kind = "ES" if context.isOpenGLES() else "Desktop"
        raise RuntimeError(f"failed to obtain functions for OpenGL {kind} {format.version()} {format.profile()}")

    return glfn

def setupStencil(glfn, drawArraysArgs):
    # on entry, VAO and Program have been bound

    # set clipping viewport
    glfn.glEnable(GLC.GL_STENCIL_TEST)
    glfn.glColorMask(False, False, False, False) # disable drawing to frame buffer
    glfn.glDepthMask(False)  # disable drawing to depth buffer
    glfn.glStencilFunc(GLC.GL_NEVER, 1, 0xFF)
    glfn.glStencilOp(GLC.GL_REPLACE, GLC.GL_KEEP, GLC.GL_KEEP)

    ## draw stencil pattern
    glfn.glStencilMask(0xFF)
    glfn.glClear(GLC.GL_STENCIL_BUFFER_BIT)
    glfn.glDrawArrays(*drawArraysArgs)

    glfn.glColorMask(True, True, True, True)
    glfn.glDepthMask(True)
    glfn.glStencilMask(0x00)
    glfn.glStencilFunc(GLC.GL_EQUAL, 1, 0xFF)

def setUniformValue(program, key, value):
    # convenience function to mask the warnings
    with warnings.catch_warnings():
        # PySide2 : RuntimeWarning: SbkConverter: Unimplemented C++ array type.
        warnings.simplefilter("ignore")
        program.setUniformValue(key, value)

class GraphicsViewGLWidget(QtOpenGLWidgets.QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self._programs = {}
        self._functions = None
        self.m_vao = QtOpenGL.QOpenGLVertexArrayObject(self)
        self.m_vbo = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.VertexBuffer)

    def initializeGL(self):
        # initializeGL gets called again when the context changes.
        # so we start off by destroying old resources.
        for program in self._programs.values():
            program.setParent(None)
        self._programs.clear()
        self.m_vao.destroy()
        self.m_vbo.destroy()
        self._functions = None

        program = QtOpenGL.QOpenGLShaderProgram()
        program.addShaderFromSourceCode(
            QtOpenGL.QOpenGLShader.ShaderTypeBit.Vertex,
            "attribute vec4 a_pos; void main() { gl_Position = a_pos; }"
        )
        program.addShaderFromSourceCode(
            QtOpenGL.QOpenGLShader.ShaderTypeBit.Fragment,
            "void main() { gl_FragColor = vec4(1.0); }"
        )
        program.bindAttributeLocation("a_pos", 0)
        program.link()
        self.storeProgram("Stencil", program)

        self.m_vao.create()
        self.m_vbo.create()

        self.m_vao.bind()
        self.m_vbo.bind()
        self.m_vbo.allocate(4 * 2 * 4)
        program.enableAttributeArray(0)
        program.setAttributeBuffer(0, GLC.GL_FLOAT, 0, 2)
        self.m_vbo.release()
        self.m_vao.release()

    def retrieveProgram(self, key):
        return self._programs.get(key)

    def storeProgram(self, key, program):
        if (olditem := self._programs.get(key)) is not None:
            olditem.setParent(None)
        program.setParent(self)
        self._programs[key] = program

    def getFunctions(self):
        if self._functions is None:
            self._functions = getFunctions(self.context())
        return self._functions

    def drawStencil(self, view):
        proj = QtGui.QMatrix4x4()
        proj.ortho(0, self.width(), self.height(), 0, -999999, 999999)
        rect = view.mapRectToScene(view.boundingRect())
        rect = proj.mapRect(rect)
        x0, y0, x1, y1 = rect.getCoords()

        buf = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], dtype=np.float32)
        self.m_vbo.bind()
        self.m_vbo.write(0, buf, buf.nbytes)
        self.m_vbo.release()

        self.retrieveProgram("Stencil").bind()
        self.m_vao.bind()
        glfn = self.getFunctions()
        setupStencil(glfn, (GLC.GL_TRIANGLE_STRIP, 0, 4))
        self.m_vao.release()
