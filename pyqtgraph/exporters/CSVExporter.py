from ..Qt import QtGui, QtCore
from .Exporter import Exporter
from ..parametertree import Parameter
from .. import PlotItem

__all__ = ['CSVExporter']
    
    
class CSVExporter(Exporter):
    Name = "CSV from plot data"
    windows = []
    def __init__(self, item):
        Exporter.__init__(self, item)
        self.params = Parameter(name='params', type='group', children=[
            {'name': 'separator', 'type': 'list', 'value': 'comma', 'values': ['comma', 'tab']},
            {'name': 'precision', 'type': 'int', 'value': 10, 'limits': [0, None]},
            {'name': 'columnMode', 'type': 'list', 'values': ['(x,y) per plot', '(x,y,y,y) for all plots']}
        ])
        
    def parameters(self):
        return self.params

    def export(self, fileName=None):

        if not isinstance(self.item, PlotItem):
            raise Exception("Must have a PlotItem selected for CSV export.")

        if fileName is None:
            self.fileSaveDialog(filter=["*.csv", "*.tsv"])
            return

        fd = open(fileName, 'w')
        data = []
        header = []

        headers = self.item.get_headers()
        new_data = []

        appendAllX = self.params['columnMode'] == '(x,y) per plot'

        for i, c in enumerate(self.item.curves):
            cd = c.getData()
            if cd[0] is None:
                continue
            data.append(cd)
            if hasattr(c, 'implements') and c.implements('plotData') and c.name() is not None:
                name = c.name().replace('"', '""') + '_'
                xName, yName = '"' + name + 'x"', '"' + name + 'y"'
            else:
                xName = headers[i][0]
                yName = headers[i][1]
            if i == 0:
                header.extend([xName, yName])
                new_data.append(cd[0])
                new_data.append(cd[1])
            else:
                header.extend([yName])
                new_data.append(cd[1])

        if self.params['separator'] == 'comma':
            sep = ','
        else:
            sep = '\t'

        fd.write(sep.join(header) + '\n')
        i = 0
        numFormat = '%%0.%dg' % self.params['precision']
        numColumns = len(header)
        numRows = len(new_data[0])

        for i in range(numRows):
            for j in range(numColumns):
                fd.write(numFormat % new_data[j][i] + sep)
            fd.write('\n')
        fd.close()

CSVExporter.register()        
                
        
