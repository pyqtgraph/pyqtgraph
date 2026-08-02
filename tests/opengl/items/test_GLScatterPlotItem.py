import pytest
pytest.importorskip('OpenGL')

import importlib

from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.opengl import GLScatterPlotItem

from common import ensure_parentItem


def test_parentItem():
    parent = GLGraphicsItem()
    child = GLScatterPlotItem(parentItem=parent)
    ensure_parentItem(parent, child)


def test_scatter_shader_program_disables_pyopengl_validation(monkeypatch):
    scatter_module = importlib.import_module("pyqtgraph.opengl.items.GLScatterPlotItem")

    class FakeFormat:
        def version(self):
            return (3, 1)

    class FakeContext:
        def format(self):
            return FakeFormat()

        def isOpenGLES(self):
            return False

    compile_program_kwargs = []

    def compile_program(*compiled, **kwargs):
        compile_program_kwargs.append(kwargs)
        return 1

    monkeypatch.setattr(GLScatterPlotItem, "_shaderProgram", None)
    monkeypatch.setattr(
        scatter_module.QtGui.QOpenGLContext,
        "currentContext",
        lambda: FakeContext(),
    )
    monkeypatch.setattr(
        scatter_module.shaders,
        "compileShader",
        lambda sources, shader_type: shader_type,
    )
    monkeypatch.setattr(scatter_module.shaders, "compileProgram", compile_program)
    monkeypatch.setattr(scatter_module.GL, "glBindAttribLocation", lambda *args: None)
    monkeypatch.setattr(scatter_module.GL, "glLinkProgram", lambda program: None)

    GLScatterPlotItem.getShaderProgram()

    assert compile_program_kwargs == [{"validate": False}]
