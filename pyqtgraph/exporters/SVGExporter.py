from .Exporter import Exporter
from pyqtgraph.parametertree import Parameter
from pyqtgraph.Qt import QtGui, QtCore, QtSvg
import pyqtgraph as pg
import re
import xml.dom.minidom as xml



__all__ = ['SVGExporter']

class SVGExporter(Exporter):
    Name = "Scalable Vector Graphics (SVG)"
    def __init__(self, item):
        Exporter.__init__(self, item)
        #tr = self.getTargetRect()
        self.params = Parameter(name='params', type='group', children=[
            #{'name': 'width', 'type': 'float', 'value': tr.width(), 'limits': (0, None)},
            #{'name': 'height', 'type': 'float', 'value': tr.height(), 'limits': (0, None)},
        ])
        #self.params.param('width').sigValueChanged.connect(self.widthChanged)
        #self.params.param('height').sigValueChanged.connect(self.heightChanged)

    def widthChanged(self):
        sr = self.getSourceRect()
        ar = sr.height() / sr.width()
        self.params.param('height').setValue(self.params['width'] * ar, blockSignal=self.heightChanged)
        
    def heightChanged(self):
        sr = self.getSourceRect()
        ar = sr.width() / sr.height()
        self.params.param('width').setValue(self.params['height'] * ar, blockSignal=self.widthChanged)
        
    def parameters(self):
        return self.params
    
    def export(self, fileName=None, toBytes=False):
        if toBytes is False and fileName is None:
            self.fileSaveDialog(filter="Scalable Vector Graphics (*.svg)")
            return
        #self.svg = QtSvg.QSvgGenerator()
        #self.svg.setFileName(fileName)
        #dpi = QtGui.QDesktopWidget().physicalDpiX()
        ### not really sure why this works, but it seems to be important:
        #self.svg.setSize(QtCore.QSize(self.params['width']*dpi/90., self.params['height']*dpi/90.))
        #self.svg.setResolution(dpi)
        ##self.svg.setViewBox()
        #targetRect = QtCore.QRect(0, 0, self.params['width'], self.params['height'])
        #sourceRect = self.getSourceRect()
        
        #painter = QtGui.QPainter(self.svg)
        #try:
            #self.setExportMode(True)
            #self.render(painter, QtCore.QRectF(targetRect), sourceRect)
        #finally:
            #self.setExportMode(False)
        #painter.end()

        ## Workaround to set pen widths correctly
        #data = open(fileName).readlines()
        #for i in range(len(data)):
            #line = data[i]
            #m = re.match(r'(<g .*)stroke-width="1"(.*transform="matrix\(([^\)]+)\)".*)', line)
            #if m is not None:
                ##print "Matched group:", line
                #g = m.groups()
                #matrix = list(map(float, g[2].split(',')))
                ##print "matrix:", matrix
                #scale = max(abs(matrix[0]), abs(matrix[3]))
                #if scale == 0 or scale == 1.0:
                    #continue
                #data[i] = g[0] + ' stroke-width="%0.2g" ' % (1.0/scale) + g[1] + '\n'
                ##print "old line:", line
                ##print "new line:", data[i]
        #open(fileName, 'w').write(''.join(data))
        
        ## Qt's SVG generator is not complete. (notably, it lacks clipping)
        ## Instead, we will use Qt to generate SVG for each item independently,
        ## then manually reconstruct the entire document.
        xml = generateSvg(self.item)
        
        if toBytes:
            return bytes(xml)
        else:
            with open(fileName, 'w') as fh:
                fh.write(xml)

xmlHeader = """\
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"  version="1.2" baseProfile="tiny">
<title>pyqtgraph SVG export</title>
<desc>Generated with Qt and pyqtgraph</desc>
<defs>
</defs>
"""

def generateSvg(item):
    global xmlHeader
    try:
        node = _generateItemSvg(item)
    finally:
        ## reset export mode for all items in the tree
        if isinstance(item, QtGui.QGraphicsScene):
            items = item.items()
        else:
            items = [item]
            for i in items:
                items.extend(i.childItems())
        for i in items:
            if hasattr(i, 'setExportMode'):
                i.setExportMode(False)
    
    cleanXml(node)
    
    return xmlHeader + node.toprettyxml(indent='    ') + "\n</svg>\n"


def _generateItemSvg(item, nodes=None, root=None):
    ## This function is intended to work around some issues with Qt's SVG generator
    ## and SVG in general.
    ## 1) Qt SVG does not implement clipping paths. This is absurd.
    ##    The solution is to let Qt generate SVG for each item independently,
    ##    then glue them together manually with clipping.
    ## 2) There seems to be wide disagreement over whether path strokes
    ##    should be scaled anisotropically. 
    ##      see: http://web.mit.edu/jonas/www/anisotropy/
    ##    Given that both inkscape and illustrator seem to prefer isotropic
    ##    scaling, we will optimize for those cases.  
    ## 3) Qt generates paths using non-scaling-stroke from SVG 1.2, but 
    ##    inkscape only supports 1.1. 
    
    
    
    if nodes is None:  ## nodes maps all node IDs to their XML element. 
                       ## this allows us to ensure all elements receive unique names.
        nodes = {}
        
    if root is None:
        root = item
        
    ## Skip hidden items
    if hasattr(item, 'isVisible') and not item.isVisible():
        return None
        
    ## If this item defines its own SVG generator, use that.
    if hasattr(item, 'generateSvg'):
        return item.generateSvg(nodes)
    

    ## Generate SVG text for just this item (exclude its children; we'll handle them later)
    tr = QtGui.QTransform()
    if isinstance(item, QtGui.QGraphicsScene):
        xmlStr = "<g>\n</g>\n"
        childs = [i for i in item.items() if i.parentItem() is None]
        doc = xml.parseString(xmlStr)
    else:
        childs = item.childItems()
        tr.translate(item.pos().x(), item.pos().y())
        tr = tr * item.transform()
        #if not item.isVisible() or int(item.flags() & item.ItemHasNoContents) > 0:
            #m = (tr.m11(), tr.m12(), tr.m21(), tr.m22(), tr.m31(), tr.m32())
            ##print item, m
            #xmlStr = '<g><g transform="matrix(%f,%f,%f,%f,%f,%f)"></g></g>' % m  # note: outer group is needed to separate clipping from transform
            #doc = xml.parseString(xmlStr)
        #else:
        arr = QtCore.QByteArray()
        buf = QtCore.QBuffer(arr)
        svg = QtSvg.QSvgGenerator()
        svg.setOutputDevice(buf)
        dpi = QtGui.QDesktopWidget().physicalDpiX()
        ### not really sure why this works, but it seems to be important:
        #self.svg.setSize(QtCore.QSize(self.params['width']*dpi/90., self.params['height']*dpi/90.))
        svg.setResolution(dpi)

        p = QtGui.QPainter()
        p.begin(svg)
        if hasattr(item, 'setExportMode'):
            item.setExportMode(True, {'painter': p})
        try:
            #p.setTransform(tr)
            item.paint(p, QtGui.QStyleOptionGraphicsItem(), None)
        finally:
            p.end()
            ## Can't do this here--we need to wait until all children have painted as well.
            ## this is taken care of in generateSvg instead.
            #if hasattr(item, 'setExportMode'):
                #item.setExportMode(False)
            
        xmlStr = str(arr)
        doc = xml.parseString(xmlStr)
        
    try:
        ## Get top-level group for this item
        g1 = doc.getElementsByTagName('g')[0]
        ## get list of sub-groups
        g2 = [n for n in g1.childNodes if isinstance(n, xml.Element) and n.tagName == 'g']
    except:
        print doc.toxml()
        raise
    
    ## make sure g1 has the transformation matrix
    m = (tr.m11(), tr.m12(), tr.m21(), tr.m22(), tr.m31(), tr.m32())
    g1.setAttribute('transform', "matrix(%f,%f,%f,%f,%f,%f)" % m)
    
    #print "=================",item,"====================="
    #print g1.toprettyxml(indent="  ", newl='')
    
    ## Inkscape does not support non-scaling-stroke (this is SVG 1.2, inkscape supports 1.1)
    ## So we need to correct anything attempting to use this.
    correctStroke(g1, item, root)
    
    ## decide on a name for this item
    baseName = item.__class__.__name__
    i = 1
    while True:
        name = baseName + "_%d" % i
        if name not in nodes:
            break
        i += 1
    nodes[name] = g1
    g1.setAttribute('id', name)
    
    ## If this item clips its children, we need to take car of that.
    childGroup = g1  ## add children directly to this node unless we are clipping
    if not isinstance(item, QtGui.QGraphicsScene):
        ## See if this item clips its children
        if int(item.flags() & item.ItemClipsChildrenToShape) > 0:
            ## Generate svg for just the path
            path = QtGui.QGraphicsPathItem(item.shape())
            pathNode = _generateItemSvg(path, root=root).getElementsByTagName('path')[0]
            ## and for the clipPath element
            clip = name + '_clip'
            clipNode = g1.ownerDocument.createElement('clipPath')
            clipNode.setAttribute('id', clip)
            clipNode.appendChild(pathNode)
            g1.appendChild(clipNode)
            
            childGroup = g1.ownerDocument.createElement('g')
            childGroup.setAttribute('clip-path', 'url(#%s)' % clip)
            g1.appendChild(childGroup)
    ## Add all child items as sub-elements.
    childs.sort(key=lambda c: c.zValue())
    for ch in childs:
        cg = _generateItemSvg(ch, nodes, root)
        if cg is None:
            continue
        childGroup.appendChild(cg)  ### this isn't quite right--some items draw below their parent (good enough for now)
    
    return g1



def correctStroke(node, item, root, width=1):
    #print "==============", item, node
    if node.hasAttribute('stroke-width'):
        width = float(node.getAttribute('stroke-width'))
    if node.getAttribute('vector-effect') == 'non-scaling-stroke':
        node.removeAttribute('vector-effect')
        if isinstance(root, QtGui.QGraphicsScene):
            w = item.mapFromScene(pg.Point(width,0))
            o = item.mapFromScene(pg.Point(0,0))
        else:
            w = item.mapFromItem(root, pg.Point(width,0))
            o = item.mapFromItem(root, pg.Point(0,0))
        w = w-o
        #print "   ", w, o, w-o
        w = (w.x()**2 + w.y()**2) ** 0.5
        #print "   ", w
        node.setAttribute('stroke-width', str(w))
    
    for ch in node.childNodes:
        if isinstance(ch, xml.Element):
            correctStroke(ch, item, root, width)
            
def cleanXml(node):
    ## remove extraneous text; let the xml library do the formatting.
    hasElement = False
    nonElement = []
    for ch in node.childNodes:
        if isinstance(ch, xml.Element):
            hasElement = True
            cleanXml(ch)
        else:
            nonElement.append(ch)
    
    if hasElement:
        for ch in nonElement:
            node.removeChild(ch)

