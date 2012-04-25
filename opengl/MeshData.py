from pyqtgraph.Qt import QtGui
import pyqtgraph.functions as fn

class MeshData(object):
    """
    Class for storing and operating on 3D mesh data. May contain:
    
    - list of vertex locations
    - list of edges
    - list of triangles
    - colors per vertex, edge, or tri
    - normals per vertex or tri
    """

    def __init__(self):
        self._vertexes = []
        self._edges = None
        self._faces = []
        self._vertexFaces = None  ## maps vertex ID to a list of face IDs
        self._vertexNormals = None
        self._faceNormals = None
        self._vertexColors = None
        self._edgeColors = None
        self._faceColors = None
        self._meshColor = (1, 1, 1, 0.1)  # default color to use if no face/edge/vertex colors are given
        
    def setFaces(self, faces, vertexes=None):
        """
        Set the faces in this data set.
        Data may be provided either as an Nx3x3 list of floats (9 float coordinate values per face)::
        
            faces = [ [(x, y, z), (x, y, z), (x, y, z)], ... ] 
            
        or as an Nx3 list of ints (vertex integers) AND an Mx3 list of floats (3 float coordinate values per vertex)::
        
            faces = [ (p1, p2, p3), ... ]
            vertexes = [ (x, y, z), ... ]
            
        """
        
        if vertexes is None:
            self._setUnindexedFaces(faces)
        else:
            self._setIndexedFaces(faces, vertexes)
    
    def setMeshColor(self, color):
        """Set the color of the entire mesh. This removes any per-face or per-vertex colors."""
        color = fn.Color(color)
        self._meshColor = color.glColor()
        self._vertexColors = None
        self._faceColors = None
    
    def _setUnindexedFaces(self, faces):
        verts = {}
        self._faces = []
        self._vertexes = []
        self._vertexFaces = []
        self._faceNormals = None
        self._vertexNormals = None
        for face in faces:
            inds = []
            for pt in face:
                pt2 = tuple([round(x*1e14) for x in pt])  ## quantize to be sure that nearly-identical points will be merged
                index = verts.get(pt2, None)
                if index is None:
                    self._vertexes.append(QtGui.QVector3D(*pt))
                    self._vertexFaces.append([])
                    index = len(self._vertexes)-1
                    verts[pt2] = index
                self._vertexFaces[index].append(len(self._faces))
                inds.append(index)
            self._faces.append(tuple(inds))
    
    def _setIndexedFaces(self, faces, vertexes):
        self._vertexes = [QtGui.QVector3D(*v) for v in vertexes]
        self._faces = faces
        self._edges = None
        self._vertexFaces = None
        self._faceNormals = None
        self._vertexNormals = None

    def vertexFaces(self):
        """
        Return list mapping each vertex index to a list of face indexes that use the vertex.
        """
        if self._vertexFaces is None:
            self._vertexFaces = [[]] * len(self._vertexes)
            for i, face in enumerate(self._faces):
                for ind in face:
                    if len(self._vertexFaces[ind]) == 0:
                        self._vertexFaces[ind] = []  ## need a unique/empty list to fill
                    self._vertexFaces[ind].append(i)
        return self._vertexFaces
        
    def __iter__(self):
        """Iterate over all faces, yielding a list of three tuples [(position, normal, color), ...] for each face."""
        vnorms = self.vertexNormals()
        vcolors = self.vertexColors()
        for i in range(len(self._faces)):
            face = []
            for j in [0,1,2]:
                vind = self._faces[i][j]
                pos = self._vertexes[vind]
                norm = vnorms[vind]
                if vcolors is None:
                    color = self._meshColor
                else:
                    color = vcolors[vind]
                face.append((pos, norm, color))
            yield face
    
    
    def faceNormals(self):
        """
        Computes and stores normal of each face.
        """
        if self._faceNormals is None:
            self._faceNormals = []
            for i, face in enumerate(self._faces):
                ## compute face normal
                pts = [self._vertexes[vind] for vind in face]
                norm = QtGui.QVector3D.crossProduct(pts[1]-pts[0], pts[2]-pts[0])
                norm = norm / norm.length()  ## don't use .normalized(); doesn't work for small values.
                self._faceNormals.append(norm)
        return self._faceNormals
    
    def vertexNormals(self):
        """
        Assigns each vertex the average of its connected face normals.
        If face normals have not been computed yet, then generateFaceNormals will be called.
        """
        if self._vertexNormals is None:
            faceNorms = self.faceNormals()
            vertFaces = self.vertexFaces()
            self._vertexNormals = []
            for vindex in xrange(len(self._vertexes)):
                #print vertFaces[vindex]
                norms = [faceNorms[findex] for findex in vertFaces[vindex]]
                norm = QtGui.QVector3D()
                for fn in norms:
                    norm += fn
                norm = norm / norm.length()  ## don't use .normalize(); doesn't work for small values.
                self._vertexNormals.append(norm)
        return self._vertexNormals
        
    def vertexColors(self):
        return self._vertexColors
        
    def faceColors(self):
        return self._faceColors
        
    def edgeColors(self):
        return self._edgeColors
        
    #def reverseNormals(self):
        #"""
        #Reverses the direction of all normal vectors.
        #"""
        #pass
        
    #def generateEdgesFromFaces(self):
        #"""
        #Generate a set of edges by listing all the edges of faces and removing any duplicates.
        #Useful for displaying wireframe meshes.
        #"""
        #pass
        
    def save(self):
        """Serialize this mesh to a string appropriate for disk storage"""
        import pickle
        names = ['_vertexes', '_edges', '_faces', '_vertexFaces', '_vertexNormals', '_faceNormals', '_vertexColors', '_edgeColors', '_faceColors', '_meshColor']
        state = {n:getattr(self, n) for n in names}
        return pickle.dumps(state)
        
    def restore(self, state):
        """Restore the state of a mesh previously saved using save()"""
        import pickle
        state = pickle.loads(state)
        for k in state:
            setattr(self, k, state[k])
        
        
        