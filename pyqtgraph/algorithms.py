import numpy as np
from .Qt import QtGui

__all__ = ['isocurve', 'isosurface', 'pseudoScatter', 'toposort']

# isocurve is used by:
# - IsocurveItem
# - functions.traceImage (which itself is unused)
# isosurface is used by:
# - GLIsosurface.py example
# pseudoScatter is used by:
# - ScatterPlotWidget
# - beeswarm.py example
# - histogram.py example
# toposort is used by:
# - Flowchart

def isocurve(data, level, connected=False, extendToEdge=False, path=False):
    """
    Generate isocurve from 2D data using marching squares algorithm.
    
    ============== =========================================================
    **Arguments:**
    data           2D numpy array of scalar values
    level          The level at which to generate an isosurface
    connected      If False, return a single long list of point pairs
                   If True, return multiple long lists of connected point 
                   locations. (This is slower but better for drawing 
                   continuous lines)
    extendToEdge   If True, extend the curves to reach the exact edges of 
                   the data. 
    path           if True, return a QPainterPath rather than a list of 
                   vertex coordinates. This forces connected=True.
    ============== =========================================================
    
    This function is SLOW; plenty of room for optimization here.
    """    
    
    if path is True:
        connected = True
    
    if extendToEdge:
        d2 = np.empty((data.shape[0]+2, data.shape[1]+2), dtype=data.dtype)
        d2[1:-1, 1:-1] = data
        d2[0, 1:-1] = data[0]
        d2[-1, 1:-1] = data[-1]
        d2[1:-1, 0] = data[:, 0]
        d2[1:-1, -1] = data[:, -1]
        d2[0,0] = d2[0,1]
        d2[0,-1] = d2[1,-1]
        d2[-1,0] = d2[-1,1]
        d2[-1,-1] = d2[-1,-2]
        data = d2
    
    sideTable = [
        [],
        [0,1],
        [1,2],
        [0,2],
        [0,3],
        [1,3],
        [0,1,2,3],
        [2,3],
        [2,3],
        [0,1,2,3],
        [1,3],
        [0,3],
        [0,2],
        [1,2],
        [0,1],
        []
        ]
    
    edgeKey=[
        [(0,1), (0,0)],
        [(0,0), (1,0)],
        [(1,0), (1,1)],
        [(1,1), (0,1)]
        ]
    
    
    lines = []
    
    ## mark everything below the isosurface level
    mask = data < level
    
    ### make four sub-fields and compute indexes for grid cells
    index = np.zeros([x-1 for x in data.shape], dtype=np.ubyte)
    fields = np.empty((2,2), dtype=object)
    slices = [slice(0,-1), slice(1,None)]
    for i in [0,1]:
        for j in [0,1]:
            fields[i,j] = mask[slices[i], slices[j]]
            #vertIndex = i - 2*j*i + 3*j + 4*k  ## this is just to match Bourk's vertex numbering scheme
            vertIndex = i+2*j
            #print i,j,k," : ", fields[i,j,k], 2**vertIndex
            np.add(index, fields[i,j] * 2**vertIndex, out=index, casting='unsafe')
            #print index
    #print index
    
    ## add lines
    for i in range(index.shape[0]):                 # data x-axis
        for j in range(index.shape[1]):             # data y-axis     
            sides = sideTable[index[i,j]]
            for l in range(0, len(sides), 2):     ## faces for this grid cell
                edges = sides[l:l+2]
                pts = []
                for m in [0,1]:      # points in this face
                    p1 = edgeKey[edges[m]][0] # p1, p2 are points at either side of an edge
                    p2 = edgeKey[edges[m]][1]
                    v1 = data[i+p1[0], j+p1[1]] # v1 and v2 are the values at p1 and p2
                    v2 = data[i+p2[0], j+p2[1]]
                    f = (level-v1) / (v2-v1)
                    fi = 1.0 - f
                    p = (    ## interpolate between corners
                        p1[0]*fi + p2[0]*f + i + 0.5, 
                        p1[1]*fi + p2[1]*f + j + 0.5
                        )
                    if extendToEdge:
                        ## check bounds
                        p = (
                            min(data.shape[0]-2, max(0, p[0]-1)),
                            min(data.shape[1]-2, max(0, p[1]-1)),                        
                        )
                    if connected:
                        gridKey = i + (1 if edges[m]==2 else 0), j + (1 if edges[m]==3 else 0), edges[m]%2
                        pts.append((p, gridKey))  ## give the actual position and a key identifying the grid location (for connecting segments)
                    else:
                        pts.append(p)
                
                lines.append(pts)

    if not connected:
        return lines
                
    ## turn disjoint list of segments into continuous lines

    #lines = [[2,5], [5,4], [3,4], [1,3], [6,7], [7,8], [8,6], [11,12], [12,15], [11,13], [13,14]]
    #lines = [[(float(a), a), (float(b), b)] for a,b in lines]
    points = {}  ## maps each point to its connections
    for a,b in lines:
        if a[1] not in points:
            points[a[1]] = []
        points[a[1]].append([a,b])
        if b[1] not in points:
            points[b[1]] = []
        points[b[1]].append([b,a])

    ## rearrange into chains
    for k in list(points.keys()):
        try:
            chains = points[k]
        except KeyError:   ## already used this point elsewhere
            continue
        #print "===========", k
        for chain in chains:
            #print "  chain:", chain
            x = None
            while True:
                if x == chain[-1][1]:
                    break ## nothing left to do on this chain
                    
                x = chain[-1][1]
                if x == k:  
                    break ## chain has looped; we're done and can ignore the opposite chain
                y = chain[-2][1]
                connects = points[x]
                for conn in connects[:]:
                    if conn[1][1] != y:
                        #print "    ext:", conn
                        chain.extend(conn[1:])
                #print "    del:", x
                del points[x]
            if chain[0][1] == chain[-1][1]:  # looped chain; no need to continue the other direction
                chains.pop()
                break
                

    ## extract point locations 
    lines = []
    for chain in points.values():
        if len(chain) == 2:
            chain = chain[1][1:][::-1] + chain[0]  # join together ends of chain
        else:
            chain = chain[0]
        lines.append([p[0] for p in chain])
    
    if not path:
        return lines ## a list of pairs of points
    
    path = QtGui.QPainterPath()
    for line in lines:
        path.moveTo(*line[0])
        for p in line[1:]:
            path.lineTo(*p)
    
    return path


IsosurfaceDataCache = None
def isosurface(data, level):
    """
    Generate isosurface from volumetric data using marching cubes algorithm.
    See Paul Bourke, "Polygonising a Scalar Field"  
    (http://paulbourke.net/geometry/polygonise/)
    
    *data*   3D numpy array of scalar values. Must be contiguous.
    *level*  The level at which to generate an isosurface
    
    Returns an array of vertex coordinates (Nv, 3) and an array of 
    per-face vertex indexes (Nf, 3)    
    """
    ## For improvement, see:
    ## 
    ## Efficient implementation of Marching Cubes' cases with topological guarantees.
    ## Thomas Lewiner, Helio Lopes, Antonio Wilson Vieira and Geovan Tavares.
    ## Journal of Graphics Tools 8(2): pp. 1-15 (december 2003)
    
    ## Precompute lookup tables on the first run
    global IsosurfaceDataCache
    if IsosurfaceDataCache is None:
        ## map from grid cell index to edge index.
        ## grid cell index tells us which corners are below the isosurface,
        ## edge index tells us which edges are cut by the isosurface.
        ## (Data stolen from Bourk; see above.)
        edgeTable = np.array([
            0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
            0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
            0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
            0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
            0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
            0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
            0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
            0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
            0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
            0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
            0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
            0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
            0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
            0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
            0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
            0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
            0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
            0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
            0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
            0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
            0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
            0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
            0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
            0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
            0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
            0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
            0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
            0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
            0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
            0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
            0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
            0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0   
            ], dtype=np.uint16)
        
        ## Table of triangles to use for filling each grid cell.
        ## Each set of three integers tells us which three edges to
        ## draw a triangle between.
        ## (Data stolen from Bourk; see above.)
        triTable = [
            [],
            [0, 8, 3],
            [0, 1, 9],
            [1, 8, 3, 9, 8, 1],
            [1, 2, 10],
            [0, 8, 3, 1, 2, 10],
            [9, 2, 10, 0, 2, 9],
            [2, 8, 3, 2, 10, 8, 10, 9, 8],
            [3, 11, 2],
            [0, 11, 2, 8, 11, 0],
            [1, 9, 0, 2, 3, 11],
            [1, 11, 2, 1, 9, 11, 9, 8, 11],
            [3, 10, 1, 11, 10, 3],
            [0, 10, 1, 0, 8, 10, 8, 11, 10],
            [3, 9, 0, 3, 11, 9, 11, 10, 9],
            [9, 8, 10, 10, 8, 11],
            [4, 7, 8],
            [4, 3, 0, 7, 3, 4],
            [0, 1, 9, 8, 4, 7],
            [4, 1, 9, 4, 7, 1, 7, 3, 1],
            [1, 2, 10, 8, 4, 7],
            [3, 4, 7, 3, 0, 4, 1, 2, 10],
            [9, 2, 10, 9, 0, 2, 8, 4, 7],
            [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4],
            [8, 4, 7, 3, 11, 2],
            [11, 4, 7, 11, 2, 4, 2, 0, 4],
            [9, 0, 1, 8, 4, 7, 2, 3, 11],
            [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1],
            [3, 10, 1, 3, 11, 10, 7, 8, 4],
            [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4],
            [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3],
            [4, 7, 11, 4, 11, 9, 9, 11, 10],
            [9, 5, 4],
            [9, 5, 4, 0, 8, 3],
            [0, 5, 4, 1, 5, 0],
            [8, 5, 4, 8, 3, 5, 3, 1, 5],
            [1, 2, 10, 9, 5, 4],
            [3, 0, 8, 1, 2, 10, 4, 9, 5],
            [5, 2, 10, 5, 4, 2, 4, 0, 2],
            [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8],
            [9, 5, 4, 2, 3, 11],
            [0, 11, 2, 0, 8, 11, 4, 9, 5],
            [0, 5, 4, 0, 1, 5, 2, 3, 11],
            [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5],
            [10, 3, 11, 10, 1, 3, 9, 5, 4],
            [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10],
            [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3],
            [5, 4, 8, 5, 8, 10, 10, 8, 11],
            [9, 7, 8, 5, 7, 9],
            [9, 3, 0, 9, 5, 3, 5, 7, 3],
            [0, 7, 8, 0, 1, 7, 1, 5, 7],
            [1, 5, 3, 3, 5, 7],
            [9, 7, 8, 9, 5, 7, 10, 1, 2],
            [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3],
            [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2],
            [2, 10, 5, 2, 5, 3, 3, 5, 7],
            [7, 9, 5, 7, 8, 9, 3, 11, 2],
            [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11],
            [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7],
            [11, 2, 1, 11, 1, 7, 7, 1, 5],
            [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11],
            [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0],
            [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0],
            [11, 10, 5, 7, 11, 5],
            [10, 6, 5],
            [0, 8, 3, 5, 10, 6],
            [9, 0, 1, 5, 10, 6],
            [1, 8, 3, 1, 9, 8, 5, 10, 6],
            [1, 6, 5, 2, 6, 1],
            [1, 6, 5, 1, 2, 6, 3, 0, 8],
            [9, 6, 5, 9, 0, 6, 0, 2, 6],
            [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8],
            [2, 3, 11, 10, 6, 5],
            [11, 0, 8, 11, 2, 0, 10, 6, 5],
            [0, 1, 9, 2, 3, 11, 5, 10, 6],
            [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11],
            [6, 3, 11, 6, 5, 3, 5, 1, 3],
            [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6],
            [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9],
            [6, 5, 9, 6, 9, 11, 11, 9, 8],
            [5, 10, 6, 4, 7, 8],
            [4, 3, 0, 4, 7, 3, 6, 5, 10],
            [1, 9, 0, 5, 10, 6, 8, 4, 7],
            [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4],
            [6, 1, 2, 6, 5, 1, 4, 7, 8],
            [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7],
            [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6],
            [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9],
            [3, 11, 2, 7, 8, 4, 10, 6, 5],
            [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11],
            [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6],
            [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6],
            [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6],
            [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11],
            [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7],
            [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9],
            [10, 4, 9, 6, 4, 10],
            [4, 10, 6, 4, 9, 10, 0, 8, 3],
            [10, 0, 1, 10, 6, 0, 6, 4, 0],
            [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10],
            [1, 4, 9, 1, 2, 4, 2, 6, 4],
            [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4],
            [0, 2, 4, 4, 2, 6],
            [8, 3, 2, 8, 2, 4, 4, 2, 6],
            [10, 4, 9, 10, 6, 4, 11, 2, 3],
            [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6],
            [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10],
            [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1],
            [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3],
            [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1],
            [3, 11, 6, 3, 6, 0, 0, 6, 4],
            [6, 4, 8, 11, 6, 8],
            [7, 10, 6, 7, 8, 10, 8, 9, 10],
            [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10],
            [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0],
            [10, 6, 7, 10, 7, 1, 1, 7, 3],
            [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7],
            [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9],
            [7, 8, 0, 7, 0, 6, 6, 0, 2],
            [7, 3, 2, 6, 7, 2],
            [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7],
            [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7],
            [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11],
            [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1],
            [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6],
            [0, 9, 1, 11, 6, 7],
            [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0],
            [7, 11, 6],
            [7, 6, 11],
            [3, 0, 8, 11, 7, 6],
            [0, 1, 9, 11, 7, 6],
            [8, 1, 9, 8, 3, 1, 11, 7, 6],
            [10, 1, 2, 6, 11, 7],
            [1, 2, 10, 3, 0, 8, 6, 11, 7],
            [2, 9, 0, 2, 10, 9, 6, 11, 7],
            [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8],
            [7, 2, 3, 6, 2, 7],
            [7, 0, 8, 7, 6, 0, 6, 2, 0],
            [2, 7, 6, 2, 3, 7, 0, 1, 9],
            [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6],
            [10, 7, 6, 10, 1, 7, 1, 3, 7],
            [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8],
            [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7],
            [7, 6, 10, 7, 10, 8, 8, 10, 9],
            [6, 8, 4, 11, 8, 6],
            [3, 6, 11, 3, 0, 6, 0, 4, 6],
            [8, 6, 11, 8, 4, 6, 9, 0, 1],
            [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6],
            [6, 8, 4, 6, 11, 8, 2, 10, 1],
            [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6],
            [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9],
            [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3],
            [8, 2, 3, 8, 4, 2, 4, 6, 2],
            [0, 4, 2, 4, 6, 2],
            [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8],
            [1, 9, 4, 1, 4, 2, 2, 4, 6],
            [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1],
            [10, 1, 0, 10, 0, 6, 6, 0, 4],
            [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3],
            [10, 9, 4, 6, 10, 4],
            [4, 9, 5, 7, 6, 11],
            [0, 8, 3, 4, 9, 5, 11, 7, 6],
            [5, 0, 1, 5, 4, 0, 7, 6, 11],
            [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5],
            [9, 5, 4, 10, 1, 2, 7, 6, 11],
            [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5],
            [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2],
            [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6],
            [7, 2, 3, 7, 6, 2, 5, 4, 9],
            [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7],
            [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0],
            [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8],
            [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7],
            [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4],
            [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10],
            [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10],
            [6, 9, 5, 6, 11, 9, 11, 8, 9],
            [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5],
            [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11],
            [6, 11, 3, 6, 3, 5, 5, 3, 1],
            [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6],
            [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10],
            [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5],
            [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3],
            [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2],
            [9, 5, 6, 9, 6, 0, 0, 6, 2],
            [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8],
            [1, 5, 6, 2, 1, 6],
            [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6],
            [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0],
            [0, 3, 8, 5, 6, 10],
            [10, 5, 6],
            [11, 5, 10, 7, 5, 11],
            [11, 5, 10, 11, 7, 5, 8, 3, 0],
            [5, 11, 7, 5, 10, 11, 1, 9, 0],
            [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1],
            [11, 1, 2, 11, 7, 1, 7, 5, 1],
            [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11],
            [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7],
            [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2],
            [2, 5, 10, 2, 3, 5, 3, 7, 5],
            [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5],
            [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2],
            [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2],
            [1, 3, 5, 3, 7, 5],
            [0, 8, 7, 0, 7, 1, 1, 7, 5],
            [9, 0, 3, 9, 3, 5, 5, 3, 7],
            [9, 8, 7, 5, 9, 7],
            [5, 8, 4, 5, 10, 8, 10, 11, 8],
            [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0],
            [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5],
            [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4],
            [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8],
            [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11],
            [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5],
            [9, 4, 5, 2, 11, 3],
            [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4],
            [5, 10, 2, 5, 2, 4, 4, 2, 0],
            [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9],
            [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2],
            [8, 4, 5, 8, 5, 3, 3, 5, 1],
            [0, 4, 5, 1, 0, 5],
            [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5],
            [9, 4, 5],
            [4, 11, 7, 4, 9, 11, 9, 10, 11],
            [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11],
            [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11],
            [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4],
            [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2],
            [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3],
            [11, 7, 4, 11, 4, 2, 2, 4, 0],
            [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4],
            [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9],
            [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7],
            [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10],
            [1, 10, 2, 8, 7, 4],
            [4, 9, 1, 4, 1, 7, 7, 1, 3],
            [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1],
            [4, 0, 3, 7, 4, 3],
            [4, 8, 7],
            [9, 10, 8, 10, 11, 8],
            [3, 0, 9, 3, 9, 11, 11, 9, 10],
            [0, 1, 10, 0, 10, 8, 8, 10, 11],
            [3, 1, 10, 11, 3, 10],
            [1, 2, 11, 1, 11, 9, 9, 11, 8],
            [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9],
            [0, 2, 11, 8, 0, 11],
            [3, 2, 11],
            [2, 3, 8, 2, 8, 10, 10, 8, 9],
            [9, 10, 2, 0, 9, 2],
            [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8],
            [1, 10, 2],
            [1, 3, 8, 9, 1, 8],
            [0, 9, 1],
            [0, 3, 8],
            []
        ]    
        edgeShifts = np.array([  ## maps edge ID (0-11) to (x,y,z) cell offset and edge ID (0-2)
            [0, 0, 0, 0],   
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [1, 0, 1, 1],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 2],
            [1, 0, 0, 2],
            [1, 1, 0, 2],
            [0, 1, 0, 2],
            #[9, 9, 9, 9]  ## fake
        ], dtype=np.uint16) # don't use ubyte here! This value gets added to cell index later; will need the extra precision.
        nTableFaces = np.array([len(f)/3 for f in triTable], dtype=np.ubyte)
        faceShiftTables = [None]
        for i in range(1,6):
            ## compute lookup table of index: vertexes mapping
            faceTableI = np.zeros((len(triTable), i*3), dtype=np.ubyte)
            faceTableInds = np.argwhere(nTableFaces == i)
            faceTableI[faceTableInds[:,0]] = np.array([triTable[j[0]] for j in faceTableInds])
            faceTableI = faceTableI.reshape((len(triTable), i, 3))
            faceShiftTables.append(edgeShifts[faceTableI])
            
        ## Let's try something different:
        #faceTable = np.empty((256, 5, 3, 4), dtype=np.ubyte)   # (grid cell index, faces, vertexes, edge lookup)
        #for i,f in enumerate(triTable):
            #f = np.array(f + [12] * (15-len(f))).reshape(5,3)
            #faceTable[i] = edgeShifts[f]
        
        
        IsosurfaceDataCache = (faceShiftTables, edgeShifts, edgeTable, nTableFaces)
    else:
        faceShiftTables, edgeShifts, edgeTable, nTableFaces = IsosurfaceDataCache

    # We use strides below, which means we need contiguous array input.
    # Ideally we can fix this just by removing the dependency on strides.
    if not data.flags['C_CONTIGUOUS']:
        raise TypeError("isosurface input data must be c-contiguous.")
    
    ## mark everything below the isosurface level
    mask = data < level
    
    ### make eight sub-fields and compute indexes for grid cells
    index = np.zeros([x-1 for x in data.shape], dtype=np.ubyte)
    fields = np.empty((2,2,2), dtype=object)
    slices = [slice(0,-1), slice(1,None)]
    for i in [0,1]:
        for j in [0,1]:
            for k in [0,1]:
                fields[i,j,k] = mask[slices[i], slices[j], slices[k]]
                vertIndex = i - 2*j*i + 3*j + 4*k  ## this is just to match Bourk's vertex numbering scheme
                np.add(index, fields[i,j,k] * 2**vertIndex, out=index, casting='unsafe')
    
    ### Generate table of edges that have been cut
    cutEdges = np.zeros([x+1 for x in index.shape]+[3], dtype=np.uint32)
    edges = edgeTable[index]
    for i, shift in enumerate(edgeShifts[:12]):        
        slices = [slice(int(shift[j]),cutEdges.shape[j]+(int(shift[j])-1)) for j in range(3)]
        cutEdges[slices[0], slices[1], slices[2], shift[3]] += edges & 2**i
    
    ## for each cut edge, interpolate to see where exactly the edge is cut and generate vertex positions
    m = cutEdges > 0
    vertexInds = np.argwhere(m)   ## argwhere is slow!
    vertexes = vertexInds[:,:3].astype(np.float32)
    dataFlat = data.reshape(data.shape[0]*data.shape[1]*data.shape[2])
    
    ## re-use the cutEdges array as a lookup table for vertex IDs
    cutEdges[vertexInds[:,0], vertexInds[:,1], vertexInds[:,2], vertexInds[:,3]] = np.arange(vertexInds.shape[0])
    
    for i in [0,1,2]:
        vim = vertexInds[:,3] == i
        vi = vertexInds[vim, :3]
        viFlat = (vi * (np.array(data.strides[:3]) // data.itemsize)[np.newaxis,:]).sum(axis=1)
        v1 = dataFlat[viFlat]
        v2 = dataFlat[viFlat + data.strides[i]//data.itemsize]
        vertexes[vim,i] += (level-v1) / (v2-v1)
    
    ### compute the set of vertex indexes for each face. 
    
    ## This works, but runs a bit slower.
    #cells = np.argwhere((index != 0) & (index != 255))  ## all cells with at least one face
    #cellInds = index[cells[:,0], cells[:,1], cells[:,2]]
    #verts = faceTable[cellInds]
    #mask = verts[...,0,0] != 9
    #verts[...,:3] += cells[:,np.newaxis,np.newaxis,:]  ## we now have indexes into cutEdges
    #verts = verts[mask]
    #faces = cutEdges[verts[...,0], verts[...,1], verts[...,2], verts[...,3]]  ## and these are the vertex indexes we want.
    
    
    ## To allow this to be vectorized efficiently, we count the number of faces in each 
    ## grid cell and handle each group of cells with the same number together.
    ## determine how many faces to assign to each grid cell
    nFaces = nTableFaces[index]
    totFaces = nFaces.sum()
    faces = np.empty((totFaces, 3), dtype=np.uint32)
    ptr = 0
    #import debug
    #p = debug.Profiler()
    
    ## this helps speed up an indexing operation later on
    cs = np.array(cutEdges.strides)//cutEdges.itemsize
    cutEdges = cutEdges.flatten()

    ## this, strangely, does not seem to help.
    #ins = np.array(index.strides)/index.itemsize
    #index = index.flatten()

    for i in range(1,6):
        ### expensive:
        #profiler()
        cells = np.argwhere(nFaces == i)  ## all cells which require i faces  (argwhere is expensive)
        #profiler()
        if cells.shape[0] == 0:
            continue
        cellInds = index[cells[:,0], cells[:,1], cells[:,2]]   ## index values of cells to process for this round
        #profiler()
        
        ### expensive:
        verts = faceShiftTables[i][cellInds]
        #profiler()
        np.add(verts[...,:3], cells[:,np.newaxis,np.newaxis,:], out=verts[...,:3], casting='unsafe')  ## we now have indexes into cutEdges
        verts = verts.reshape((verts.shape[0]*i,)+verts.shape[2:])
        #profiler()
        
        ### expensive:
        verts = (verts * cs[np.newaxis, np.newaxis, :]).sum(axis=2)
        vertInds = cutEdges[verts]
        #profiler()
        nv = vertInds.shape[0]
        #profiler()
        faces[ptr:ptr+nv] = vertInds #.reshape((nv, 3))
        #profiler()
        ptr += nv
        
    return vertexes, faces


def pseudoScatter(data, spacing=None, shuffle=True, bidir=False, method='exact'):
    """Return an array of position values needed to make beeswarm or column scatter plots.
    
    Used for examining the distribution of values in an array.
    
    Given an array of x-values, construct an array of y-values such that an x,y scatter-plot
    will not have overlapping points (it will look similar to a histogram).
    """
    if method == 'exact':
        return _pseudoScatterExact(data, spacing=spacing, shuffle=shuffle, bidir=bidir)
    elif method == 'histogram':
        return _pseudoScatterHistogram(data, spacing=spacing, shuffle=shuffle, bidir=bidir)


def _pseudoScatterHistogram(data, spacing=None, shuffle=True, bidir=False):
    """Works by binning points into a histogram and spreading them out to fill the bin.
    
    Faster method, but can produce blocky results.
    """
    inds = np.arange(len(data))
    if shuffle:
        np.random.shuffle(inds)
        
    data = data[inds]
    
    if spacing is None:
        spacing = 2.*np.std(data)/len(data)**0.5

    yvals = np.empty(len(data))
    
    dmin = data.min()
    dmax = data.max()
    nbins = int((dmax-dmin) / spacing) + 1
    bins = np.linspace(dmin, dmax, nbins)
    dx = bins[1] - bins[0]
    dbins = ((data - bins[0]) / dx).astype(int)
    binCounts = {}
        
    for i,j in enumerate(dbins):
        c = binCounts.get(j, -1) + 1
        binCounts[j] = c
        yvals[i] = c

    if bidir is True:
        for i in range(nbins):
            yvals[dbins==i] -= binCounts.get(i, 0) * 0.5

    return yvals[np.argsort(inds)]  ## un-shuffle values before returning


def _pseudoScatterExact(data, spacing=None, shuffle=True, bidir=False):
    """Works by stacking points up one at a time, searching for the lowest position available at each point.
    
    This method produces nice, smooth results but can be prohibitively slow for large datasets.
    """
    inds = np.arange(len(data))
    if shuffle:
        np.random.shuffle(inds)
        
    data = data[inds]
    
    if spacing is None:
        spacing = 2.*np.std(data)/len(data)**0.5
    s2 = spacing**2
    
    yvals = np.empty(len(data))
    if len(data) == 0:
        return yvals
    yvals[0] = 0
    for i in range(1,len(data)):
        x = data[i]     # current x value to be placed
        x0 = data[:i]   # all x values already placed
        y0 = yvals[:i]  # all y values already placed
        y = 0
        
        dx = (x0-x)**2  # x-distance to each previous point
        xmask = dx < s2  # exclude anything too far away
        
        if xmask.sum() > 0:
            if bidir:
                dirs = [-1, 1]
            else:
                dirs = [1]
            yopts = []
            for direction in dirs:
                y = 0
                dx2 = dx[xmask]
                dy = (s2 - dx2)**0.5   
                limits = np.empty((2,len(dy)))  # ranges of y-values to exclude
                limits[0] = y0[xmask] - dy
                limits[1] = y0[xmask] + dy    
                while True:
                    # ignore anything below this y-value
                    if direction > 0:
                        mask = limits[1] >= y
                    else:
                        mask = limits[0] <= y
                        
                    limits2 = limits[:,mask]
                    
                    # are we inside an excluded region?
                    mask = (limits2[0] < y) & (limits2[1] > y)
                    if mask.sum() == 0:
                        break
                        
                    if direction > 0:
                        y = limits2[:,mask].max()
                    else:
                        y = limits2[:,mask].min()
                yopts.append(y)
            if bidir:
                y = yopts[0] if -yopts[0] < yopts[1] else yopts[1]
            else:
                y = yopts[0]
        yvals[i] = y
    
    return yvals[np.argsort(inds)]  ## un-shuffle values before returning


def toposort(deps, nodes=None, seen=None, stack=None, depth=0):
    """Topological sort. Arguments are:
      deps    dictionary describing dependencies where a:[b,c] means "a depends on b and c"
      nodes   optional, specifies list of starting nodes (these should be the nodes 
              which are not depended on by any other nodes). Other candidate starting
              nodes will be ignored.
              
    Example::

        # Sort the following graph:
        # 
        #   B ──┬─────> C <── D
        #       │       │       
        #   E <─┴─> A <─┘
        #     
        deps = {'a': ['b', 'c'], 'c': ['b', 'd'], 'e': ['b']}
        toposort(deps)
         => ['b', 'd', 'c', 'a', 'e']
    """
    # fill in empty dep lists
    deps = deps.copy()
    for k,v in list(deps.items()):
        for k in v:
            if k not in deps:
                deps[k] = []
    
    if nodes is None:
        ## run through deps to find nodes that are not depended upon
        rem = set()
        for dep in deps.values():
            rem |= set(dep)
        nodes = set(deps.keys()) - rem
    if seen is None:
        seen = set()
        stack = []
    sorted = []
    for n in nodes:
        if n in stack:
            raise Exception("Cyclic dependency detected", stack + [n])
        if n in seen:
            continue
        seen.add(n)
        sorted.extend( toposort(deps, deps[n], seen, stack+[n], depth=depth+1))
        sorted.append(n)
    return sorted
