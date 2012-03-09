from OpenGL.GL import *
from OpenGL.GL import shaders

## For centralizing and managing vertex/fragment shader programs.


Shaders = {
    'balloon': (   ## increases fragment alpha as the normal turns orthogonal to the view
        """
        varying vec3 normal;
        void main() {
            normal = normalize(gl_NormalMatrix * gl_Normal);
            //vec4 color = normal;
            //normal.w = min(color.w + 2.0 * color.w * pow(normal.x*normal.x + normal.y*normal.y, 2.0), 1.0);
            gl_FrontColor = gl_Color;
            gl_BackColor = gl_Color;
            gl_Position = ftransform();
        }
        """,
        """
        varying vec3 normal;
        void main() {
            vec4 color = gl_Color;
            color.w = min(color.w + 2.0 * color.w * pow(normal.x*normal.x + normal.y*normal.y, 5.0), 1.0);
            gl_FragColor = color;
        }
        """
    ),
}
CompiledShaders = {}
    
def getShader(name):
    global Shaders, CompiledShaders
    
    if name not in CompiledShaders:
        vshader, fshader = Shaders[name]
        vcomp = shaders.compileShader(vshader, GL_VERTEX_SHADER)
        fcomp = shaders.compileShader(fshader, GL_FRAGMENT_SHADER)
        prog = shaders.compileProgram(vcomp, fcomp)
        CompiledShaders[name] = prog, vcomp, fcomp
    return CompiledShaders[name][0]
