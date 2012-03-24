class MeshData(object):
    """
    Class for storing 3D mesh data. May contain:
        - list of vertex locations
        - list of edges
        - list of triangles
        - colors per vertex, edge, or tri
        - normals per vertex or tri
    """

    def __init__(self):
        self.vertexes = []
        self.edges = None
        self.faces = []
        self.vertexFaces = None  ## maps vertex ID to a list of face IDs
        self.vertexNormals = None
        self.faceNormals = None
        self.vertexColors = None
        self.edgeColors = None
        self.faceColors = None
        
    def setFaces(self, faces, vertexes=None):
        """
        Set the faces in this data set.
        Data may be provided either as an Nx3x3 list of floats (9 float coordinate values per face)
            *faces* = [ [(x, y, z), (x, y, z), (x, y, z)], ... ] 
        or as an Nx3 list of ints (vertex integers) AND an Mx3 list of floats (3 float coordinate values per vertex)
            *faces* = [ (p1, p2, p3), ... ]
            *vertexes* = [ (x, y, z), ... ]
        """
        
        if vertexes is None:
            self._setUnindexedFaces(self, faces)
        else:
            self._setIndexedFaces(self, faces)
            
    def _setUnindexedFaces(self, faces):
        verts = {}
        self.faces = []
        self.vertexes = []
        self.vertexFaces = []
        self.faceNormals = None
        self.vertexNormals = None
        for face in faces:
            inds = []
            for pt in face:
                pt2 = tuple([int(x*1e14) for x in pt])  ## quantize to be sure that nearly-identical points will be merged
                index = verts.get(pt2, None)
                if index is None:
                    self.vertexes.append(tuple(pt))
                    self.vertexFaces.append([])
                    index = len(self.vertexes)-1
                    verts[pt2] = index
                self.vertexFaces[index].append(face)
                inds.append(index)
            self.faces.append(tuple(inds))
    
    def _setIndexedFaces(self, faces, vertexes):
        self.vertexes = vertexes
        self.faces = faces
        self.edges = None
        self.vertexFaces = None
        self.faceNormals = None
        self.vertexNormals = None

    def getVertexFaces(self):
        """
        Return list mapping each vertex index to a list of face indexes that use the vertex.
        """
        if self.vertexFaces is None:
            self.vertexFaces = [[]] * len(self.vertexes)
            for i, face in enumerate(self.faces):
                for ind in face:
                    if len(self.vertexFaces[ind]) == 0:
                        self.vertexFaces[ind] = []  ## need a unique/empty list to fill
                    self.vertexFaces[ind].append(i)
        return self.vertexFaces
        
        
    def getFaceNormals(self):
        """
        Computes and stores normal of each face.
        """
        if self.faceNormals is None:
            self.faceNormals = []
            for i, face in enumerate(self.faces):
                ## compute face normal
                pts = [QtGui.QVector3D(*self.vertexes[vind]) for vind in face]
                norm = QtGui.QVector3D.crossProduct(pts[1]-pts[0], pts[2]-pts[0])
                self.faceNormals.append(norm)
        return self.faceNormals
    
    def getVertexNormals(self):
        """
        Assigns each vertex the average of its connected face normals.
        If face normals have not been computed yet, then generateFaceNormals will be called.
        """
        if self.vertexNormals is None:
            faceNorms = self.getFaceNormals()
            vertFaces = self.getVertexFaces()
            self.vertexNormals = []
            for vindex in xrange(len(self.vertexes)):
                norms = [faceNorms[findex] for findex in vertFaces[vindex]]
                if len(norms) == 0:
                    norm = QtGui.QVector3D()
                else:
                    norm = reduce(QtGui.QVector3D.__add__, facenorms) / float(len(norms))
                self.vertexNormals.append(norm)
        return self.vertexNormals
        
        
    def reverseNormals(self):
        """
        Reverses the direction of all normal vectors.
        """
        pass
        
    def generateEdgesFromFaces(self):
        """
        Generate a set of edges by listing all the edges of faces and removing any duplicates.
        Useful for displaying wireframe meshes.
        """
        pass
        
