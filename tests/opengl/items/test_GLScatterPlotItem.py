import sys

import pytest
pytest.importorskip('OpenGL')

from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.opengl import GLScatterPlotItem

from common import ensure_parentItem


def test_parentItem():
    parent = GLGraphicsItem()
    child = GLScatterPlotItem(parentItem=parent)
    ensure_parentItem(parent, child)


def test_scatter_shader_program_disables_pyopengl_validation(monkeypatch):
    scatter_module = sys.modules[GLScatterPlotItem.__module__]

    class FakeFormat:
        def version(self):
            return (3, 1)

    class FakeContext:
        def format(self):
            return FakeFormat()

        def isOpenGLES(self):
            return False

    class FakeQOpenGLContext:
        @staticmethod
        def currentContext():
            return FakeContext()

    compile_program_calls = []

    monkeypatch.setattr(GLScatterPlotItem, "_shaderProgram", None)
    monkeypatch.setattr(scatter_module.QtGui, "QOpenGLContext", FakeQOpenGLContext)
    monkeypatch.setattr(
        scatter_module.shaders,
        "compileShader",
        lambda sources, shader_type: (sources, shader_type),
    )

    def compile_program(*compiled, validate=True):
        compile_program_calls.append(validate)
        return 1

    monkeypatch.setattr(scatter_module.shaders, "compileProgram", compile_program)
    monkeypatch.setattr(
        scatter_module.GL, "glBindAttribLocation", lambda *args: None
    )
    monkeypatch.setattr(scatter_module.GL, "glLinkProgram", lambda *args: None)

    GLScatterPlotItem.getShaderProgram()

    assert compile_program_calls == [False]
