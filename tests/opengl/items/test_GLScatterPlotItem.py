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
    scatter_module = importlib.import_module("pyqtgraph.opengl.items.GLScatterPlotItem")
    program = 123
    compile_program_calls = []
    bound_locations = []
    linked_programs = []

    class Format:
        def version(self):
            return (3, 1)

    class Context:
        def format(self):
            return Format()

        def isOpenGLES(self):
            return False

    class OpenGLContext:
        @staticmethod
        def currentContext():
            return Context()

    def compile_shader(sources, shader_type):
        return shader_type

    def compile_program(*compiled, **kwds):
        compile_program_calls.append((compiled, kwds))
        return program

    def bind_attrib_location(program, index, name):
        bound_locations.append((program, index, name))

    def link_program(program):
        linked_programs.append(program)

    monkeypatch.setattr(scatter_module.GLScatterPlotItem, "_shaderProgram", None)
    monkeypatch.setattr(scatter_module.QtGui, "QOpenGLContext", OpenGLContext)
    monkeypatch.setattr(scatter_module.shaders, "compileShader", compile_shader)
    monkeypatch.setattr(scatter_module.shaders, "compileProgram", compile_program)
    monkeypatch.setattr(scatter_module.GL, "glBindAttribLocation", bind_attrib_location)
    monkeypatch.setattr(scatter_module.GL, "glLinkProgram", link_program)

    assert GLScatterPlotItem.getShaderProgram() == program
    assert compile_program_calls == [
        (
            (
                scatter_module.GL.GL_VERTEX_SHADER,
                scatter_module.GL.GL_FRAGMENT_SHADER,
            ),
            {"validate": False},
        )
    ]
    assert bound_locations == [
        (program, 0, "a_position"),
        (program, 1, "a_color"),
        (program, 2, "a_size"),
    ]
    assert linked_programs == [program]
