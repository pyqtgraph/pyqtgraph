import textwrap

from ..Qt import QtOpenGL

## For centralizing and managing vertex/fragment shader programs.

def initShaders():
    global Shaders
    Shaders = [
        ShaderProgram("default", [
            VertexShader(textwrap.dedent("""
                uniform mat4 u_mvp;
                attribute vec4 a_position;
                attribute vec4 a_color;
                varying vec4 v_color;
                void main() {
                    v_color = a_color;
                    gl_Position = u_mvp * a_position;
                }
            """)),
            FragmentShader(textwrap.dedent("""
                #ifdef GL_ES
                precision mediump float;
                #endif
                varying vec4 v_color;
                void main() {
                    gl_FragColor = v_color;
                }
            """))
        ]),

        ## increases fragment alpha as the normal turns orthogonal to the view
        ## this is useful for viewing shells that enclose a volume (such as isosurfaces)
        ShaderProgram('balloon', [
            VertexShader(textwrap.dedent("""
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
            """)),
            FragmentShader(textwrap.dedent("""
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
            """))
        ]),

        ## colors fragments based on face normals relative to view
        ## This means that the colors will change depending on how the view is rotated
        ShaderProgram('viewNormalColor', [   
            VertexShader(textwrap.dedent("""
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
            """)),
            FragmentShader(textwrap.dedent("""
                #ifdef GL_ES
                precision mediump float;
                #endif
                varying vec4 v_color;
                varying vec3 v_normal;
                void main() {
                    vec3 rgb = (v_normal + 1.0) * 0.5;
                    gl_FragColor = vec4(rgb, v_color.a);
                }
            """))
        ]),

        ## colors fragments based on absolute face normals.
        ShaderProgram('normalColor', [   
            VertexShader(textwrap.dedent("""
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
            """)),
            FragmentShader(textwrap.dedent("""
                #ifdef GL_ES
                precision mediump float;
                #endif
                varying vec4 v_color;
                varying vec3 v_normal;
                void main() {
                    vec3 rgb = (v_normal + 1.0) * 0.5;
                    gl_FragColor = vec4(rgb, v_color.a);
                }
            """))
        ]),

        ## very simple simulation of lighting. 
        ## The light source position is always relative to the camera.
        ShaderProgram('shaded', [   
            VertexShader(textwrap.dedent("""
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
            """)),
            FragmentShader(textwrap.dedent("""
                #ifdef GL_ES
                precision mediump float;
                #endif
                uniform float lightDirection[3];
                varying vec4 v_color;
                varying vec3 v_normal;
                void main() {
                    vec3 dirn = vec3(lightDirection[0], lightDirection[1], lightDirection[2]);
                    float p = dot(v_normal, normalize(dirn));
                    p = p < 0. ? 0. : p * 0.8;
                    vec3 rgb = v_color.rgb * (0.2 + p);
                    gl_FragColor = vec4(rgb, v_color.a);
                }
            """)),
        ], uniforms={'lightDirection': [1.0, -1.0, -1.0]}),

        ## colors get brighter near edges of object
        ShaderProgram('edgeHilight', [   
            VertexShader(textwrap.dedent("""
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
            """)),
            FragmentShader(textwrap.dedent("""
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
            """))
        ]),

        ## colors fragments by z-value.
        ## This is useful for coloring surface plots by height.
        ## This shader uses a uniform called "colorMap" to determine how to map the colors:
        ##    red   = pow(colorMap[0]*(z + colorMap[1]), colorMap[2])
        ##    green = pow(colorMap[3]*(z + colorMap[4]), colorMap[5])
        ##    blue  = pow(colorMap[6]*(z + colorMap[7]), colorMap[8])
        ## (set the values like this: shader['uniformMap'] = array([...])
        ShaderProgram('heightColor', [
            VertexShader(textwrap.dedent("""
                uniform mat4 u_mvp;
                attribute vec4 a_position;
                varying float zpos;
                void main() {
                    zpos = a_position.z;
                    gl_Position = u_mvp * a_position;
                }
            """)),
            FragmentShader(textwrap.dedent("""
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
            """)),
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

    def sourceCode(self) -> str:
        return self._sourceCode

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
        self.uniformData = {}

        ## parse extra options from the shader definition
        if uniforms is not None:
            self.uniformData.update(uniforms)

    def setUniformData(self, uniformName, data):
        if data is None:
            del self.uniformData[uniformName]
        else:
            self.uniformData[uniformName] = data

    def __setitem__(self, item, val):
        self.setUniformData(item, val)

    def __delitem__(self, item):
        self.setUniformData(item, None)

initShaders()
