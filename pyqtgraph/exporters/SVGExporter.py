from .Exporter import Exporter
from pyqtgraph.parametertree import Parameter
from pyqtgraph.Qt import QtGui, QtCore, QtSvg
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

        node = self.generateItemSvg(self.item)
        xml = """\
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"  version="1.2" baseProfile="tiny">
<title>pyqtgraph SVG export</title>
<desc>Generated with Qt and pyqtgraph</desc>
<defs>
</defs>
"""   + node.toprettyxml(indent='    ') + "\n</svg>\n"

        if toBytes:
            return bytes(xml)
        else:
            with open(fileName, 'w') as fh:
                fh.write(xml)

    def generateItemSvg(self, item):    
        if isinstance(item, QtGui.QGraphicsScene):
            xmlStr = "<g></g>"
            childs = [i for i in item.items() if i.parentItem() is None]
        else:
            tr = QtGui.QTransform()
            tr.translate(item.pos().x(), item.pos().y())
            tr = tr * item.transform()
            if not item.isVisible() or int(item.flags() & item.ItemHasNoContents) > 0:
                m = (tr.m11(), tr.m12(), tr.m21(), tr.m22(), tr.m31(), tr.m32())
                #print item, m
                xmlStr = '<g transform="matrix(%f,%f,%f,%f,%f,%f)"></g>' % m
            else:
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
                    #tr = QtGui.QTransform()
                    #tr.translate(item.pos().x(), item.pos().y())
                    #p.setTransform(tr * item.transform())
                    p.setTransform(tr)
                    item.paint(p, QtGui.QStyleOptionGraphicsItem(), None)
                finally:
                    p.end()
                    if hasattr(item, 'setExportMode'):
                        item.setExportMode(False)
                    
                xmlStr = str(arr)
            childs = item.childItems()

        doc = xml.parseString(xmlStr)
        try:
            groups = doc.getElementsByTagName('g')
            if len(groups) == 1:
                g1 = g2 = groups[0]
            else:
                g1,g2 = groups[:2]
        except:
            print doc.toxml()
            raise
        g1.setAttribute('id', item.__class__.__name__)
        
        ## Check for item visibility
        visible = True
        if not isinstance(item, QtGui.QGraphicsScene):
            parent = item
            while visible and parent is not None:
                visible = parent.isVisible()
                parent = parent.parentItem()
            
            if not visible:
                style = g1.getAttribute('style').strip()
                if len(style)>0 and not style.endswith(';'):
                    style += ';'
                style += 'display:none;'
                g1.setAttribute('style', style)
                
        childs.sort(key=lambda c: c.zValue())
        for ch in childs:
            cg = self.generateItemSvg(ch)
            g2.appendChild(cg)
        
        return g1



  ### To check:
  ###   do all items really generate this double-group structure?
  ###   are both groups necessary?
  ###   How do we implement clipping? (can we clip to an object that is visible?)







