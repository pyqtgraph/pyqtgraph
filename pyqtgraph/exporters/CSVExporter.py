import csv
import itertools

import numpy as np

from .. import ErrorBarItem, PlotItem
from ..parametertree import Parameter
from ..Qt import QtCore
from .Exporter import Exporter

translate = QtCore.QCoreApplication.translate

__all__ = ['CSVExporter']
    
    
class CSVExporter(Exporter):
    Name = "CSV of original plot data"
    windows = []
    def __init__(self, item):
        Exporter.__init__(self, item)
        self.params = Parameter(name='params', type='group', children=[
            {'name': 'separator', 'title': translate("Exporter", 'separator'), 'type': 'list', 'value': 'comma', 'limits': ['comma', 'tab']},
            {'name': 'precision', 'title': translate("Exporter", 'precision'), 'type': 'int', 'value': 10, 'limits': [0, None]},
            {
                'name': 'columnMode',
                'title': translate("Exporter", 'columnMode'),
                'type': 'list',
                'limits': ['(x,y) per plot', '(x,y,y,y) for all plots']
            }
        ])

    def parameters(self):
        return self.params

    def export(self, fileName=None):
        
        if not isinstance(self.item, PlotItem):
            raise TypeError("Must have a PlotItem selected for CSV export.")

        if fileName is None:
            self.fileSaveDialog(filter=["*.csv", "*.tsv"])
            return

        data = []
        header = []

        appendAllX = self.params['columnMode'] == '(x,y) per plot'

        # grab curve information
        for i, c in enumerate(self.item.curves):
            if hasattr(c, 'getOriginalDataset'): # try to access unmapped, unprocessed data
                cd = c.getOriginalDataset()
            else:
                cd = c.getData() # fall back to earlier access method
            if cd[0] is None:
                continue
            data.append(cd)
            if hasattr(c, 'implements') and c.implements('plotData') and c.name() is not None:
                name = c.name().replace('"', '""') + '_'
                xName = f"{name}x"
                yName = f"{name}y"
            else:
                xName = 'x%04d' % i
                yName = 'y%04d' % i

            if appendAllX or i == 0:
                header.extend([xName, yName])
            else:
                header.extend([yName])

        header_naming_map = {
            "left": "x_min_error",
            "right": "x_max_error",
            "bottom": "y_min_error",
            "top": "y_max_error"
        }

        # if there is an error-bar item grab the error bar info
        for i, c in enumerate(self.item.items):
            if not isinstance(c, ErrorBarItem):
                continue
            error_data = []
            for error_direction, header_label in header_naming_map.items():
                if (error := c.opts[error_direction]) is not None:
                    header.extend([f'{header_label}_{i:04}'])
                    error_data.append(error)
            if error_data:
                data.append(tuple(error_data))
        sep = "," if self.params['separator'] == 'comma' else "\t"

        # we want to flatten the nested arrays of data into columns
        columns = [column for dataset in data for column in dataset]
        with open(fileName, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=sep, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)
            for row in itertools.zip_longest(*columns, fillvalue=""):
                row_to_write = [
                    item if isinstance(item, str) 
                    else np.format_float_positional(
                        item, precision=self.params['precision']
                    )
                    for item in row
                ]
                writer.writerow(row_to_write)

CSVExporter.register()
