import numpy as np

from ..Qt import QtOpenGL
from ..Qt import OpenGLHelpers

## For centralizing and managing vertex/fragment shader programs.

def initShaders():
    global Shaders
    Shaders = [
        ShaderProgram(None, [
            VertexShader("""
                uniform mat4 u_mvp;
                attribute vec4 a_position;
                attribute vec4 a_color;
                varying vec4 v_color;
                void main() {
                    v_color = a_color;
                    gl_Position = u_mvp * a_position;
                }
            """),
            FragmentShader("""
                #ifdef GL_ES
                precision mediump float;
                #endif
                varying vec4 v_color;
                void main() {
                    gl_FragColor = v_color;
                }
            """)
        ]),

        ## increases fragment alpha as the normal turns orthogonal to the view
        ## this is useful for viewing shells that enclose a volume (such as isosurfaces)
        ShaderProgram('balloon', [
            VertexShader("""
                uniform mat4 u_mvp;
                uniform mat3 u_normal;
                attribute vec4 a_position;
                attribute vec3 a_normal;
                attribute vec4 a_color;
                varying vec4 v_color;
                varying vec3 v_normal;
                void main() {
                    v_normal = normalize(u_normal * a_normal);
                    v_color = a_color;
                    gl_Position = u_mvp * a_position;
                }
            """),
            FragmentShader("""
                #ifdef GL_ES
                precision mediump float;
                #endif
                varying vec4 v_color;
                varying vec3 v_normal;
                void main() {
                    vec4 color = v_color;
                    color.w = min(color.w + 2.0 * color.w * pow(v_normal.x*v_normal.x + v_normal.y*v_normal.y, 5.0), 1.0);
                    gl_FragColor = color;
                }
            """)
        ]),
        
        ## colors fragments based on face normals relative to view
        ## This means that the colors will change depending on how the view is rotated
        ShaderProgram('viewNormalColor', [   
            VertexShader("""
                uniform mat4 u_mvp;
                uniform mat3 u_normal;
                attribute vec4 a_position;
                attribute vec3 a_normal;
                attribute vec4 a_color;
                varying vec4 v_color;
                varying vec3 v_normal;
                void main() {
                    v_normal = normalize(u_normal * a_normal);
                    v_color = a_color;
                    gl_Position = u_mvp * a_position;
                }
            """),
            FragmentShader("""
                #ifdef GL_ES
                precision mediump float;
                #endif
                varying vec4 v_color;
                varying vec3 v_normal;
                void main() {
                    vec3 rgb = (v_normal + 1.0) * 0.5;
                    gl_FragColor = vec4(rgb, v_color.a);
                }
            """)
        ]),
        
        ## colors fragments based on absolute face normals.
        ShaderProgram('normalColor', [   
            VertexShader("""
                uniform mat4 u_mvp;
                attribute vec4 a_position;
                attribute vec3 a_normal;
                attribute vec4 a_color;
                varying vec4 v_color;
                varying vec3 v_normal;
                void main() {
                    v_normal = normalize(a_normal);
                    v_color = a_color;
                    gl_Position = u_mvp * a_position;
                }
            """),
            FragmentShader("""
                #ifdef GL_ES
                precision mediump float;
                #endif
                varying vec4 v_color;
                varying vec3 v_normal;
                void main() {
                    vec3 rgb = (v_normal + 1.0) * 0.5;
                    gl_FragColor = vec4(rgb, v_color.a);
                }
            """)
        ]),
        
        ## very simple simulation of lighting. 
        ## The light source position is always relative to the camera.
        ShaderProgram('shaded', [   
            VertexShader("""
                uniform mat4 u_mvp;
                uniform mat3 u_normal;
                attribute vec4 a_position;
                attribute vec3 a_normal;
                attribute vec4 a_color;
                varying vec4 v_color;
                varying vec3 v_normal;
                void main() {
                    v_normal = normalize(u_normal * a_normal);
                    v_color = a_color;
                    gl_Position = u_mvp * a_position;
                }
            """),
            FragmentShader("""
                #ifdef GL_ES
                precision mediump float;
                #endif
                varying vec4 v_color;
                varying vec3 v_normal;
                void main() {
                    float p = dot(v_normal, normalize(vec3(1.0, -1.0, -1.0)));
                    p = p < 0. ? 0. : p * 0.8;
                    vec3 rgb = v_color.rgb * (0.2 + p);
                    gl_FragColor = vec4(rgb, v_color.a);
                }
            """)
        ]),
        
        ## colors get brighter near edges of object
        ShaderProgram('edgeHilight', [   
            VertexShader("""
                uniform mat4 u_mvp;
                uniform mat3 u_normal;
                attribute vec4 a_position;
                attribute vec3 a_normal;
                attribute vec4 a_color;
                varying vec4 v_color;
                varying vec3 v_normal;
                void main() {
                    v_normal = normalize(u_normal * a_normal);
                    v_color = a_color;
                    gl_Position = u_mvp * a_position;
                }
            """),
            FragmentShader("""
                #ifdef GL_ES
                precision mediump float;
                #endif
                varying vec4 v_color;
                varying vec3 v_normal;
                void main() {
                    float s = pow(v_normal.x*v_normal.x + v_normal.y*v_normal.y, 2.0);
                    vec3 rgb = v_color.rgb + s * (1.0-v_color.rgb);
                    gl_FragColor = vec4(rgb, v_color.a);
                }
            """)
        ]),
        
        ## colors fragments by z-value.
        ## This is useful for coloring surface plots by height.
        ## This shader uses a uniform called "colorMap" to determine how to map the colors:
        ##    red   = pow(colorMap[0]*(z + colorMap[1]), colorMap[2])
        ##    green = pow(colorMap[3]*(z + colorMap[4]), colorMap[5])
        ##    blue  = pow(colorMap[6]*(z + colorMap[7]), colorMap[8])
        ## (set the values like this: shader['uniformMap'] = array([...])
        ShaderProgram('heightColor', [
            VertexShader("""
                uniform mat4 u_mvp;
                attribute vec4 a_position;
                varying float zpos;
                void main() {
                    zpos = a_position.z;
                    gl_Position = u_mvp * a_position;
                }
            """),
            FragmentShader("""
                #ifdef GL_ES
                precision mediump float;
                #endif
                uniform float colorMap[9];
                varying float zpos;
                void main() {
                    vec3 color;

                    color.x = colorMap[0] * (zpos + colorMap[1]);
                    if (colorMap[2] != 1.0)
                        color.x = pow(color.x, colorMap[2]);
                    color.x = clamp(color.x, 0.0, 1.0);
                    
                    color.y = colorMap[3] * (zpos + colorMap[4]);
                    if (colorMap[5] != 1.0)
                        color.y = pow(color.y, colorMap[5]);
                    color.y = clamp(color.y, 0.0, 1.0);
                    
                    color.z = colorMap[6] * (zpos + colorMap[7]);
                    if (colorMap[8] != 1.0)
                        color.z = pow(color.z, colorMap[8]);
                    color.z = clamp(color.z, 0.0, 1.0);
                    
                    gl_FragColor = vec4(color, 1.0);
                }
            """),
        ], uniforms={'colorMap': [1, 1, 1, 1, 0.5, 1, 1, 0, 1]}),

    ]


def getShaderProgram(name):
    return ShaderProgram.names[name]

class Shader:
    def __init__(self, shaderType: QtOpenGL.QOpenGLShader.ShaderTypeBit, sourceCode: str):
        self._shaderType : QtOpenGL.QOpenGLShader.ShaderTypeBit = shaderType
        self._sourceCode : str = sourceCode

    def shaderType(self) -> QtOpenGL.QOpenGLShader.ShaderTypeBit:
        return self._shaderType

    def sourceCode(self, *, es2_compat=False) -> str:
        """Return the source code for this shader, optionally modified for ES2 compatibility."""
        source = self._sourceCode
        if es2_compat and not source.lstrip().startswith("#version"):
            # we know that macOS OpenGL 4.1 Core has ARB_ES2_compatibility,
            # so we can get it to run legacy shaders by marking the
            # shaders as ES2.
            # The explicit #undefs counteract QOpenGLShader::compileSourceCode,
            # which predefines lowp/mediump/highp as empty macros when compiling
            # for desktop OpenGL; that would mangle the precision statements
            # inside "#ifdef GL_ES" blocks activated by "#version 100".
            source = (
                "#version 100\n"
                "#undef lowp\n#undef mediump\n#undef highp\n"
                + source
            )
        return source

class VertexShader(Shader):
    def __init__(self, sourceCode):
        super().__init__(QtOpenGL.QOpenGLShader.ShaderTypeBit.Vertex, sourceCode)

class FragmentShader(Shader):
    def __init__(self, sourceCode):
        super().__init__(QtOpenGL.QOpenGLShader.ShaderTypeBit.Fragment, sourceCode)

class ShaderProgram:
    names = {}

    def __init__(self, name, shaders, uniforms=None):
        self.name = name
        ShaderProgram.names[name] = self
        self.shaders = shaders
        self.prog : QtOpenGL.QOpenGLShaderProgram | None = None
        self.uniformData = {}
        self.glUniform1fv = None

        ## parse extra options from the shader definition
        if uniforms is not None:
            for k,v in uniforms.items():
                self[k] = v

    def setUniformData(self, uniformName, data):
        if data is None:
            del self.uniformData[uniformName]
        else:
            self.uniformData[uniformName] = data

    def __setitem__(self, item, val):
        self.setUniformData(item, val)

    def __delitem__(self, item):
        self.setUniformData(item, None)

    def program(self, *, es2_compat=False) -> QtOpenGL.QOpenGLShaderProgram:
        if self.prog is None:
            program = QtOpenGL.QOpenGLShaderProgram()
            for shader in self.shaders:
                if not program.addShaderFromSourceCode(shader.shaderType(), shader.sourceCode(es2_compat=es2_compat)):
                    raise RuntimeError("Shader compile failure:\n%s" % program.log())

            # for reasons that may vary across drivers, having vertex attribute
            # array generic location 0 enabled (glEnableVertexAttribArray(0)) is
            # required for rendering to take place.
            # this only becomes an issue if we are using glVertexAttrib{1,4}f
            # because that's when we *don't* call glEnableVertexAttribArray.
            # since we always need vertex coordinates to come from arrays, it is
            # sufficient for us to bind "a_position" explicitly to 0.
            program.bindAttributeLocation("a_position", 0)
            if not program.link():
                raise RuntimeError("Program link failure:\n%s" % program.log())
            self.prog = program
        return self.prog

    def __enter__(self):
        if (program := self.program()) is not None:
            program.bind()

            try:
                ## load uniform values into program
                for uniformName, data in self.uniformData.items():
                    if (loc := program.uniformLocation(uniformName)) == -1:
                        raise RuntimeError(f'Could not find uniform variable "{uniformName}"')

                    # we would like to use program.setUniformValueArray(), but:
                    # - PyQt has a buggy binding of setUniformValueArray for array of float.
                    #   this is still true as of PyQt6 6.11
                    # - PySide6 had a bug PYSIDE-3005 which got fixed since PySide6 >= 6.9.
                    # - In conda forge's build of PySide6, setUniformValueArray does not
                    #   accept ndarrays. Use either Python list or ctypes array

                    data = np.ascontiguousarray(data, dtype=np.float32)
                    if self.glUniform1fv is None:
                        self.glUniform1fv = OpenGLHelpers.get_gl_uniform_1fv()
                    self.glUniform1fv(loc, data.size, data.ctypes.data)

            except:
                program.release()
                raise

    def __exit__(self, *args):
        if self.prog is not None:
            self.prog.release()

    def uniform(self, name):
        """Return the location integer for a uniform variable in this program"""
        return self.program().uniformLocation(name)

initShaders()
