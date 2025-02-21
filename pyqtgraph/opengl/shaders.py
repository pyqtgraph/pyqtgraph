from OpenGL import GL
from OpenGL.GL import shaders  # noqa
try:
    from OpenGL import NullFunctionError
except ImportError:
    from OpenGL.error import NullFunctionError
import numpy as np
import re

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


CompiledShaderPrograms = {}
    
def getShaderProgram(name):
    return ShaderProgram.names[name]

class Shader(object):
    def __init__(self, shaderType, code):
        self.shaderType = shaderType
        self.code = code
        self.compiled = None
        
    def shader(self):
        if self.compiled is None:
            try:
                self.compiled = shaders.compileShader(self.code, self.shaderType)
            except NullFunctionError:
                raise Exception("This OpenGL implementation does not support shader programs; many OpenGL features in pyqtgraph will not work.")
            except RuntimeError as exc:
                ## Format compile errors a bit more nicely
                if len(exc.args) == 3:
                    err, code, typ = exc.args
                    if not err.startswith('Shader compile failure'):
                        raise
                    code = code[0].decode('utf_8').split('\n')
                    err, c, msgs = err.partition(':')
                    err = err + '\n'
                    msgs = re.sub('b\'','',msgs)
                    msgs = re.sub('\'$','',msgs)
                    msgs = re.sub('\\\\n','\n',msgs)
                    msgs = msgs.split('\n')
                    errNums = [()] * len(code)
                    for i, msg in enumerate(msgs):
                        msg = msg.strip()
                        if msg == '':
                            continue
                        m = re.match(r'(\d+\:)?\d+\((\d+)\)', msg)
                        if m is not None:
                            line = int(m.groups()[1])
                            errNums[line-1] = errNums[line-1] + (str(i+1),)
                            #code[line-1] = '%d\t%s' % (i+1, code[line-1])
                        err = err + "%d %s\n" % (i+1, msg)
                    errNums = [','.join(n) for n in errNums]
                    maxlen = max(map(len, errNums))
                    code = [errNums[i] + " "*(maxlen-len(errNums[i])) + line for i, line in enumerate(code)]
                    err = err + '\n'.join(code)
                    raise Exception(err)
                else:
                    raise
        return self.compiled

class VertexShader(Shader):
    def __init__(self, code):
        Shader.__init__(self, GL.GL_VERTEX_SHADER, code)
        
class FragmentShader(Shader):
    def __init__(self, code):
        Shader.__init__(self, GL.GL_FRAGMENT_SHADER, code)
        
        
        

class ShaderProgram(object):
    names = {}
    
    def __init__(self, name, shaders, uniforms=None):
        self.name = name
        ShaderProgram.names[name] = self
        self.shaders = shaders
        self.prog = None
        self.blockData = {}
        self.uniformData = {}
        
        ## parse extra options from the shader definition
        if uniforms is not None:
            for k,v in uniforms.items():
                self[k] = v
        
    def setBlockData(self, blockName, data):
        if data is None:
            del self.blockData[blockName]
        else:
            self.blockData[blockName] = data

    def setUniformData(self, uniformName, data):
        if data is None:
            del self.uniformData[uniformName]
        else:
            self.uniformData[uniformName] = data
            
    def __setitem__(self, item, val):
        self.setUniformData(item, val)
        
    def __delitem__(self, item):
        self.setUniformData(item, None)

    def program(self, *, es2_compat=False):
        # for reasons that may vary across drivers, having vertex attribute
        # array generic location 0 enabled (glEnableVertexAttribArray(0)) is
        # required for rendering to take place.
        # this only becomes an issue if we are using glVertexAttrib{1,4}f
        # because that's when we *don't* call glEnableVertexAttribArray.
        # since we always need vertex coordinates to come from arrays, it is
        # sufficient for us to bind "a_position" explicitly to 0.
        if self.prog is None:
            try:
                # we know that macOS OpenGL 4.1 Core has ARB_ES2_compatibility,
                # so we can get it to run legacy shaders by marking the shaders
                # as ES2
                compiled = []
                for shader in self.shaders:
                    sources = [shader.code]
                    if es2_compat and not shader.code.lstrip().startswith("#version"):
                        sources.insert(0, "#version 100\n")
                    compiled.append(shaders.compileShader(sources, shader.shaderType))
                self.prog = shaders.compileProgram(*compiled)  ## compile program
            except:
                self.prog = -1
                raise
            # bind generic vertex attrib 0 to "a_position" and relink
            GL.glBindAttribLocation(self.prog, 0, "a_position")
            GL.glLinkProgram(self.prog)
        return self.prog
        
    def __enter__(self):
        if len(self.shaders) > 0 and self.program() != -1:
            GL.glUseProgram(self.program())
            
            try:
                ## load uniform values into program
                for uniformName, data in self.uniformData.items():
                    loc = self.uniform(uniformName)
                    if loc == -1:
                        raise Exception('Could not find uniform variable "%s"' % uniformName)
                    GL.glUniform1fv(loc, len(data), np.array(data, dtype=np.float32))
                    
                ### bind buffer data to program blocks
                #if len(self.blockData) > 0:
                    #bindPoint = 1
                    #for blockName, data in self.blockData.items():
                        ### Program should have a uniform block declared:
                        ### 
                        ### layout (std140) uniform blockName {
                        ###     vec4 diffuse;
                        ### };
                        
                        ### pick any-old binding point. (there are a limited number of these per-program
                        #bindPoint = 1
                        
                        ### get the block index for a uniform variable in the shader
                        #blockIndex = glGetUniformBlockIndex(self.program(), blockName)
                        
                        ### give the shader block a binding point
                        #glUniformBlockBinding(self.program(), blockIndex, bindPoint)
                        
                        ### create a buffer
                        #buf = glGenBuffers(1)
                        #glBindBuffer(GL_UNIFORM_BUFFER, buf)
                        #glBufferData(GL_UNIFORM_BUFFER, size, data, GL_DYNAMIC_DRAW)
                        ### also possible to use glBufferSubData to fill parts of the buffer
                        
                        ### bind buffer to the same binding point
                        #glBindBufferBase(GL_UNIFORM_BUFFER, bindPoint, buf)
            except:
                GL.glUseProgram(0)
                raise
                    
            
        
    def __exit__(self, *args):
        if len(self.shaders) > 0:
            GL.glUseProgram(0)
        
    def uniform(self, name):
        """Return the location integer for a uniform variable in this program"""
        return GL.glGetUniformLocation(self.program(), name.encode('utf_8'))

    #def uniformBlockInfo(self, blockName):
        #blockIndex = glGetUniformBlockIndex(self.program(), blockName)
        #count = glGetActiveUniformBlockiv(self.program(), blockIndex, GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS)
        #indices = []
        #for i in range(count):
            #indices.append(glGetActiveUniformBlockiv(self.program(), blockIndex, GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES))

initShaders()
