class MeshData(object):
    """
    Class for storing 3D mesh data. May contain:
        - list of vertex locations
        - list of edges
        - list of triangles
        - colors per vertex, edge, or tri
        - normals per vertex or tri
    """

    def __init__(self ...):


    def generateFaceNormals(self):
        
    
    def generateVertexNormals(self):
        """
        Assigns each vertex the average of its connected face normals.
        If face normals have not been computed yet, then generateFaceNormals will be called.
        """
        
        
    def reverseNormals(self):
    