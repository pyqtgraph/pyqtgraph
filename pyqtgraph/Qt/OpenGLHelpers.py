import importlib

from . import QT_LIB, QtWidgets, QtVersionInfo
from . import OpenGLConstants as GLC

if QtVersionInfo[0] >= 6:
    QtOpenGLWidgets = importlib.import_module(f"{QT_LIB}.QtOpenGLWidgets")
else:
    QtOpenGLWidgets = QtWidgets

__all__ = ["getFunctions", "GraphicsViewGLWidget"]

def getFunctions(context):
    glfn = None
    format = context.format()

    if QT_LIB.startswith("PySide"):
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

class GraphicsViewGLWidget(QtOpenGLWidgets.QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self._programs = {}
        self._functions = None

    def initializeGL(self):
        # initializeGL gets called again when the context changes.
        # so we start off by destroying old resources.
        for program in self._programs.values():
            program.setParent(None)
        self._programs.clear()
        self._functions = None

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

    def setViewboxClip(self, view):
        rect = view.sceneBoundingRect()
        dpr = self.devicePixelRatioF()
        # glScissor wants the bottom-left corner and is Y-up
        x, y = rect.left(), self.height() - rect.bottom()
        w, h = rect.width(), rect.height()
        glfn = self.getFunctions()
        glfn.glScissor(*[round(v * dpr) for v in [x, y, w, h]])
        glfn.glEnable(GLC.GL_SCISSOR_TEST)
        # the test will be disabled by QPainter.endNativePainting().
