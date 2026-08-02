import importlib

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
    scatter = importlib.import_module("pyqtgraph.opengl.items.GLScatterPlotItem")

    class FakeFormat:
        def version(self):
            return (4, 6)

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
    attrib_locations = []
    linked_programs = []
    program = 123

    def fake_compile_shader(sources, shader_type):
        assert sources
        return shader_type

    def fake_compile_program(*compiled, **kwargs):
        compile_program_calls.append((compiled, kwargs))
        return program

    def fake_gl_bind_attrib_location(program_arg, location, name):
        attrib_locations.append((program_arg, location, name))

    def fake_gl_link_program(program_arg):
        linked_programs.append(program_arg)

    monkeypatch.setattr(GLScatterPlotItem, "_shaderProgram", None)
    monkeypatch.setattr(scatter.QtGui, "QOpenGLContext", FakeQOpenGLContext)
    monkeypatch.setattr(scatter.shaders, "compileShader", fake_compile_shader)
    monkeypatch.setattr(scatter.shaders, "compileProgram", fake_compile_program)
    monkeypatch.setattr(scatter.GL, "glBindAttribLocation", fake_gl_bind_attrib_location)
    monkeypatch.setattr(scatter.GL, "glLinkProgram", fake_gl_link_program)

    assert GLScatterPlotItem.getShaderProgram() == program
    assert compile_program_calls == [((scatter.GL.GL_VERTEX_SHADER, scatter.GL.GL_FRAGMENT_SHADER), {"validate": False})]
    assert attrib_locations == [
        (program, 0, "a_position"),
        (program, 1, "a_color"),
        (program, 2, "a_size"),
    ]
    assert linked_programs == [program]
