# -*- coding: utf-8 -*-
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
            {'name': 'separator', 'type': 'list', 'value': 'comma', 'values': ['comma', 'tab', 'align']},
            {'name': 'precision', 'type': 'int', 'value': 10, 'limits': [0, None]},
            {'name': 'columnMode', 'type': 'list', 'values': ['(x,y) per plot', '(x,y,y,y) for all plots']}
        ])
        
    def parameters(self):
        return self.params
    
    def export(self, fileName=None):

        if not isinstance(self.item, PlotItem):
            raise Exception("Must have a PlotItem selected for CSV export.")
        
        if fileName is None:
            self.fileSaveDialog(filter=["*.csv", "*.tsv", "*.mv", "*.dat"])
            return

        data = []
        header = []

        appendAllX = self.params['columnMode'] == '(x,y) per plot'

        for i, c in enumerate(self.item.curves):
            cd = c.getData()
            if cd[0] is None:
                continue
            data.append(cd)
            if hasattr(c, 'implements') and c.implements('plotData') and c.name() is not None:
                name = c.name().replace('"', '""') + '_'
                xName, yName = '"'+name+'x"', '"'+name+'y"'
            elif hasattr(c, 'xname') and hasattr(c, 'yname'):
                xName = c.xname
                yName = c.yname
            else:
                xName = 'x%04d' % i
                yName = 'y%04d' % i
            if appendAllX or i == 0:
                header.extend([xName, yName])
            else:
                header.extend([yName])

        sep = None
        if self.params['separator'] == 'comma':
            sep = ','
        elif self.params['separator'] == 'tab':
            sep = '\t'
        elif self.params['separator'] == 'align':
            sep = ' '

        # CSV or Tabular
        if sep is not ' ':
            with open(fileName, 'w') as fd:
                fd.write(sep.join(header) + '\n')
                i = 0
                numFormat = '%%0.%dg' % self.params['precision']
                numRows = max([len(d[0]) for d in data])
                for i in range(numRows):
                    for j, d in enumerate(data):
                        # write x value if this is the first column, or if we want
                        # x for all rows
                        if appendAllX or j == 0:
                            if d is not None and i < len(d[0]):
                                fd.write(numFormat % d[0][i] + sep)
                            else:
                                fd.write(' %s' % sep)
    
                        # write y value
                        if d is not None and i < len(d[1]):
                            fd.write(numFormat % d[1][i] + sep)
                        else:
                            fd.write(' %s' % sep)
                    fd.write('\n')
        # Aligned with spaces
        else:
            width = self.params['precision']
            max_header_width = 0
            for col in header:
                if len(col) > max_header_width:
                    max_header_width = len(col)
            exp_width = 4 + 2 # 1.{width}E+00
            total_width = max_header_width + exp_width
            with open(fileName, 'w') as fd:
                header_write = ''
                for col in header:
                   header_write += '{str:<{width}.{width}s}{sep}'.format(str=col,width=max_header_width,sep=sep)
                fd.write(header_write + '\n')
                i = 0
                numRows = max([len(d[0]) for d in data])
                for i in range(numRows):
                    for j, d in enumerate(data):
                        # write x value if this is the first column, or if we want
                        # x for all rows
                        if appendAllX or j == 0:
                            if d is not None and i < len(d[0]):
                                line_write = '{num:<{width}.{precision}E}{sep}'.format(num=d[0][i],width=max_header_width,precision=width,sep=sep)
                                fd.write(line_write)
                            else:
                                fd.write(' %s' % sep)
    
                        # write y value
                        if d is not None and i < len(d[1]):
                            line_write = '{num:<{width}.{precision}E}{sep}'.format(num=d[1][i],width=max_header_width,precision=width,sep=sep)
                            fd.write(line_write)
                        else:
                            fd.write(' %s' % sep)
                    fd.write('\n')

CSVExporter.register()        
                
        
