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
        self.params = Parameter.create(name='params', type='group', children=[
            {'name': 'separator', 'title': translate("Exporter", 'separator'), 'type': 'list', 'value': 'comma', 'limits': ['comma', 'tab']},
            {'name': 'precision', 'title': translate("Exporter", 'precision'), 'type': 'int', 'value': 10, 'limits': [0, None]},
            {
                'name': 'columnMode',
                'title': translate("Exporter", 'columnMode'),
                'type': 'list',
                'limits': ['(x,y) per plot', '(x,y,y,y) for all plots'],
                'value': '(x,y) per plot',
            }
        ])

        self.index_counter = itertools.count(start=0)
        self.header = []
        self.data = []

    def parameters(self):
        return self.params

    def _exportErrorBarItem(self, errorBarItem: ErrorBarItem) -> None:
        error_data = []
        index = next(self.index_counter)

        # make sure the plot actually has data:
        if errorBarItem.opts['x'] is None or errorBarItem.opts['y'] is None:
            return None

        header_naming_map = {
            "left": "x_min_error",
            "right": "x_max_error",
            "bottom": "y_min_error",
            "top": "y_max_error"
        }

        # grab the base-points
        self.header.extend([f'x{index:04}_error', f'y{index:04}_error'])
        error_data.extend([errorBarItem.opts['x'], errorBarItem.opts['y']])

        # grab the error bars
        for error_direction, header_label in header_naming_map.items():
            if (error := errorBarItem.opts[error_direction]) is not None:
                self.header.extend([f'{header_label}_{index:04}'])
                error_data.append(error)

        self.data.append(tuple(error_data))
        return None

    def _exportPlotDataItem(self, plotDataItem) -> None:
        if hasattr(plotDataItem, 'getOriginalDataset'):
            # try to access unmapped, unprocessed data
            cd = plotDataItem.getOriginalDataset()
        else:
             # fall back to earlier access method
            cd = plotDataItem.getData()
        if cd[0] is None:
            # no data found, break out...
            return None
        self.data.append(cd)

        index = next(self.index_counter)
        if plotDataItem.name() is not None:
            name = plotDataItem.name().replace('"', '""') + '_'
            xName = f"{name}x"
            yName = f"{name}y"
        else:
            xName = f'x{index:04}'
            yName = f'y{index:04}'
        appendAllX = self.params['columnMode'] == '(x,y) per plot'
        if appendAllX or index == 0:
            self.header.extend([xName, yName])
        else:
            self.header.extend([yName])
        return None

    def export(self, fileName=None):
        if not isinstance(self.item, PlotItem):
            raise TypeError("Must have a PlotItem selected for CSV export.")

        if fileName is None:
            self.fileSaveDialog(filter=["*.csv", "*.tsv"])
            return

        for item in self.item.items:
            if isinstance(item, ErrorBarItem):
                self._exportErrorBarItem(item)
            elif hasattr(item, 'implements') and item.implements('plotData'):
                self._exportPlotDataItem(item)

        sep = "," if self.params['separator'] == 'comma' else "\t"
        # we want to flatten the nested arrays of data into columns
        columns = [column for dataset in self.data for column in dataset]
        with open(fileName, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=sep, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(self.header)
            for row in itertools.zip_longest(*columns, fillvalue=""):
                row_to_write = [
                    item if isinstance(item, str) 
                    else np.format_float_positional(
                        item, precision=self.params['precision']
                    )
                    for item in row
                ]
                writer.writerow(row_to_write)

        self.header.clear()
        self.data.clear()

CSVExporter.register()
