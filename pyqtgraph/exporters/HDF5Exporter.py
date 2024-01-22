import importlib.util

import numpy

from .. import PlotItem
from ..parametertree import Parameter
from ..Qt import QtCore
from .Exporter import Exporter

HAVE_HDF5 = importlib.util.find_spec("h5py") is not None

translate = QtCore.QCoreApplication.translate

__all__ = ['HDF5Exporter']

    
class HDF5Exporter(Exporter):
    Name = "HDF5 Export: plot (x,y)"
    windows = []
    allowCopy = False

    def __init__(self, item):
        Exporter.__init__(self, item)
        self.params = Parameter.create(name='params', type='group', children=[
            {'name': 'Name', 'title': translate("Exporter", 'Name'), 'type': 'str', 'value': 'Export', },
            {'name': 'columnMode', 'title': translate("Exporter", 'columnMode'), 'type': 'list',
             'limits': ['(x,y) per plot', '(x,y,y,y) for all plots'], 'value': '(x,y) per plot'},
        ])
        
    def parameters(self):
        return self.params
    
    def export(self, fileName=None):
        if not HAVE_HDF5:
            raise RuntimeError("This exporter requires the h5py package, "
                               "but it was not importable.")
        
        import h5py

        if not isinstance(self.item, PlotItem):
            raise Exception("Must have a PlotItem selected for HDF5 export.")
        
        if fileName is None:
            self.fileSaveDialog(filter=["*.h5", "*.hdf", "*.hd5"])
            return
        dsname = self.params['Name']
        fd = h5py.File(fileName, 'a')  # forces append to file... 'w' doesn't seem to "delete/overwrite"
        data = []

        appendAllX = self.params['columnMode'] == '(x,y) per plot'
        # Check if the arrays are ragged
        len_first = len(self.item.curves[0].getData()[0]) if self.item.curves[0] else None
        ragged = any(len(i.getData()[0]) != len_first for i in self.item.curves)

        if ragged:
            dgroup = fd.create_group(dsname)
            for i, c in enumerate(self.item.curves):
                d = c.getData()
                fdata = numpy.array([d[0], d[1]]).astype('double')
                cname = c.name() if c.name() is not None else str(i)
                dgroup.create_dataset(cname, data=fdata)
        else:
            for i, c in enumerate(self.item.curves):
                d = c.getData()
                if appendAllX or i == 0:
                    data.append(d[0])
                data.append(d[1])

            fdata = numpy.array(data).astype('double')
            fd.create_dataset(dsname, data=fdata)

        fd.close()

if HAVE_HDF5:
    HDF5Exporter.register()
