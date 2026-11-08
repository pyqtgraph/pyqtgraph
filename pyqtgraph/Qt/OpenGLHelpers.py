import ctypes
import importlib
import sys

from . import QT_LIB, QtCore, QtGui, QtWidgets, QtOpenGL, QtVersionInfo
from . import OpenGLConstants as GLC
from . import compat

if QtVersionInfo[0] >= 6:
    QtOpenGLWidgets = importlib.import_module(f"{QT_LIB}.QtOpenGLWidgets")
else:
    QtOpenGLWidgets = QtWidgets

__all__ = ["getFunctions", "GraphicsViewGLWidget"]

# for use with glDrawElements
NULL = compat.voidptr(0) if QT_LIB.startswith("PySide") else None

# OpenGL library is stdcall on Windows
GLFUNCTYPE = ctypes.WINFUNCTYPE if sys.platform == "win32" else ctypes.CFUNCTYPE

GLsizei = ctypes.c_int
GLuint = ctypes.c_uint
GLenum = ctypes.c_uint

GLDRAWELEMENTS_TYPE = GLFUNCTYPE(
    None,
    GLenum,     # mode
    GLsizei,    # count
    GLenum,     # type
    ctypes.c_void_p
)

GLPRIMITIVERESTARTINDEX_TYPE = GLFUNCTYPE(None, GLuint)

OFFSET_PTR = lambda x : compat.voidptr(x, 0, False)

def getDrawElements(ctx, glfn):
    if QT_LIB.startswith("PySide"):
        return glfn.glDrawElements, OFFSET_PTR
    elif QtCore.PYQT_VERSION >= 0x60800:
        return glfn.glDrawElements, OFFSET_PTR
    else:
        # PyQt < (6, 8)'s binding of glDrawElements crashes if
        # a voidptr is passed in, so we will load it manually.
        if not hasattr(glfn, '_glDrawElements_ctypes'):
            proc_addr = ctx.getProcAddress(b"glDrawElements")
            glfn._glDrawElements_ctypes = GLDRAWELEMENTS_TYPE(int(proc_addr))
        return glfn._glDrawElements_ctypes, ctypes.c_void_p

def getPrimitiveRestart(ctx, glfn):
    is_opengles = ctx.isOpenGLES()
    gl_version = ctx.format().version()
    if gl_version >= ((3, 0) if is_opengles else (4, 3)):
        primitive_restart = GLC.GL_PRIMITIVE_RESTART_FIXED_INDEX
    elif not is_opengles and gl_version >= (3, 1):
        if not hasattr(glfn, 'glPrimitiveRestartIndex'):
            # this branch occurs for the following conditions:
            # - PySide and (3, 1) <= gl_version <= (4, 2)
            #   - because extraFunctions() does not include glPrimitiveRestartIndex
            # - PyQt   and (3, 1) <= gl_version <= (4, 0)
            #   - because PyQt only provides bindings for 2.1 and 4.1 Core
            proc_addr = ctx.getProcAddress(b"glPrimitiveRestartIndex")
            glfn.glPrimitiveRestartIndex = GLPRIMITIVERESTARTINDEX_TYPE(int(proc_addr))
        glfn.glPrimitiveRestartIndex(0xFFFFFFFF)
        primitive_restart = GLC.GL_PRIMITIVE_RESTART
    else:
        raise RuntimeError("OpenGL Primitive Restart not supported")
    return primitive_restart

def getFunctions(context) -> QtOpenGL.QAbstractOpenGLFunctions:
    glfn = None
    format = context.format()

    if QT_LIB.startswith("PySide"):
        glfn = context.extraFunctions()

    elif not context.isOpenGLES() and QT_LIB.startswith("PyQt") and QtVersionInfo >= (6, 0):
        # VersionFunctionsFactory doesn't support ES
        vp = QtOpenGL.QOpenGLVersionProfile()
        if format.version() >= (4, 1):
            vp.setVersion(4, 1)
            vp.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        else:
            vp.setVersion(2, 1)
        glfn = QtOpenGL.QOpenGLVersionFunctionsFactory.get(vp, context)

    else:
        # PyQt5 has context.versionFunctions().
        #    however, when there are multiple GraphicsItems, the following bug occurs:
        #    all except one of the C++ objects of the returned versionFunctions() get
        #    deleted. i.e. in PyQt5, we are not able to cache the return value.
        # To overcome the above listed issues, we load the modules directly.

        # Note: The GraphicsItems now all share the same context held in
        #       GraphicsViewGLWidget, so technically we aren't affected by the not
        #       being able to cache issue.

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
            if not glfn.initializeOpenGLFunctions():
                glfn = None
                continue
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


GLUNIFORM1FV_TYPE = GLFUNCTYPE(
    None,             # Return type (void)
    ctypes.c_int,     # location
    ctypes.c_int,     # count
    ctypes.c_void_p
)

def get_gl_uniform_1fv(context):
    func_ptr = context.getProcAddress(b"glUniform1fv")
    return GLUNIFORM1FV_TYPE(int(func_ptr))

def upload_vbo(vbo: QtOpenGL.QOpenGLBuffer, arr) -> None:
    if arr is None:
        vbo.destroy()
        return
    if not vbo.isCreated():
        vbo.create()
    vbo.bind()
    if vbo.size() != arr.nbytes:
        vbo.allocate(arr, arr.nbytes)
    else:
        vbo.write(0, arr, arr.nbytes)
    vbo.release()

def compile_and_link(context, *,
                     sources_core, sources_legacy,
                     attributes: dict[str, int]
                    ):

    fmt = context.format()

    if sources_core is None:
        # we only have legacy shaders for GLMeshItem

        if context.hasExtension(b'GL_ARB_ES2_compatibility'):
            # we know that macOS OpenGL 4.1 Core has ARB_ES2_compatibility,
            # so we can get it to run legacy shaders by marking the
            # shaders as ES2.
            # The explicit #undefs counteract QOpenGLShader::compileSourceCode,
            # which predefines lowp/mediump/highp as empty macros when compiling
            # for desktop OpenGL; that would mangle the precision statements
            # inside "#ifdef GL_ES" blocks activated by "#version 100".
            prefix = (
                "#version 100\n"
                "#undef lowp\n#undef mediump\n#undef highp\n"
            )
        else:
            # if unmarked,
            #   OpenGL Desktop treats it as #version 110
            #   OpenGL ES treats it as #version 100
            prefix = ""
        sources = sources_legacy

    elif context.isOpenGLES():
        if fmt.version() >= (3, 0):
            prefix = "#version 300 es\n"
            sources = sources_core
        else:
            prefix = "#version 100\n"
            sources = sources_legacy
    else:
        if fmt.version() >= (3, 1):
            prefix = "#version 140\n"
            sources = sources_core
        else:
            prefix = "#version 120\n"   # OpenGL 2.1
            sources = sources_legacy

    program = QtOpenGL.QOpenGLShaderProgram()
    for shader_type, src in sources.items():
        if not program.addShaderFromSourceCode(shader_type, prefix + src):
            raise RuntimeError(program.log())

    # for reasons that may vary across drivers, having vertex attribute
    # array generic location 0 enabled (glEnableVertexAttribArray(0)) is
    # required for rendering to take place.
    # this only becomes an issue if we are using glVertexAttrib{1,4}f
    # because that's when we *don't* call glEnableVertexAttribArray.
    # since we always need vertex coordinates to come from arrays, it is
    # sufficient for us to bind "a_position" explicitly to 0.
    for name, pos in attributes.items():
        program.bindAttributeLocation(name, pos)

    if not program.link():
        raise RuntimeError(program.log())

    return program
