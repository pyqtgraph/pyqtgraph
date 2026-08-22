import importlib.util

import numpy

from .. import PlotItem, ViewBox
from ..parametertree import Parameter
from ..GraphicsScene import GraphicsScene
from ..Qt import QtCore
from .Exporter import Exporter

HAVE_HDF5 = importlib.util.find_spec("h5py") is not None

translate = QtCore.QCoreApplication.translate

__all__ = ["HDF5Exporter"]


class HDF5Exporter(Exporter):
    Name = "HDF5 Export: plot (x,y)"
    windows = []
    allowCopy = False

    def __init__(self, item):
        Exporter.__init__(self, item)
        self.params = Parameter.create(
            name=translate("Exporter", "params"),
            type="group",
            children=[
                {
                    "name": "columnMode",
                    "title": translate("Exporter", "columnMode"),
                    "type": "list",
                    "limits": [
                        translate("Exporter", "2d (x,y) per plot"),
                        translate("Exporter", "separate x and y"),
                    ],
                    "value": translate("Exporter", "2d (x,y) per plot"),
                },
                {
                    "name": "originalDataset",
                    "title": translate("Exporter", "originalDataset"),
                    "type": "bool",
                    "value": True,
                },
            ],
        )

    def parameters(self):
        return self.params

    def export(self, fileName=None):
        if not HAVE_HDF5:
            raise RuntimeError(
                "This exporter requires the h5py package, "
                "but it was not importable."
            )

        import h5py

        item = self.item
        items = []

        if isinstance(item, GraphicsScene):
            # why this order???
            items = [
                _item
                for _item in reversed(item.items())
                if isinstance(_item, PlotItem)
            ]
        if isinstance(item, ViewBox):
            items = [
                _item
                for _item in [item.parentItem()]
                if isinstance(_item, PlotItem)
            ]
        if isinstance(item, PlotItem):
            items = [item]
        if not items:
            raise Exception("Must have a PlotItem for HDF5 export.")

        if fileName is None:
            self.fileSaveDialog(
                # see https://www.hdfgroup.org/solutions/hdf5/
                filter=["*.h5", "*.hdf5"],
            )
            return

        appendAllX = self.params["columnMode"] == translate(
            "Exporter", "2d (x,y) per plot"
        )

        with h5py.File(fileName, "w") as fd:
            for index, item in enumerate(items, start=1):
                pd = fd if len(items) == 1 else fd.require_group(
                    item.titleLabel.text
                    if item.titleLabel.isVisible()
                    else f"Plot {index}"
                )
                if appendAllX:
                    for i, c in enumerate(item.curves, start=1):
                        d = (
                            c.getOriginalDataset()
                            if self.params["originalDataset"]
                            else c.getData()
                        )
                        if d[0] is None or d[1] is None:
                            continue
                        fdata = numpy.column_stack(d)
                        cname = c.name() or f"Curve {i}"
                        pd.require_dataset(
                            name=cname,
                            shape=fdata.shape,
                            dtype=fdata.dtype,
                            data=fdata,
                        )
                else:
                    x_axis = item.getAxis("bottom")
                    x_label = (
                        f"{x_axis.labelText} ({x_axis.labelUnits})"
                        if x_axis.labelUnits
                        else x_axis.labelText or "x"
                    )
                    y_axis = item.getAxis("left")
                    y_label = (
                        f"{y_axis.labelText} ({y_axis.labelUnits})"
                        if y_axis.labelUnits
                        else y_axis.labelText or "y"
                    )

                    for i, c in enumerate(item.curves, start=1):
                        d = (
                            c.getOriginalDataset()
                            if self.params["originalDataset"]
                            else c.getData()
                        )
                        cname = c.name() or f"Curve {i}"
                        cg = pd.require_group(cname)
                        if d[0] is not None:
                            cg.require_dataset(
                                name=x_label,
                                shape=d[0].shape,
                                dtype=d[0].dtype,
                                data=d[0],
                            )
                        if d[1] is not None:
                            cg.require_dataset(
                                name=y_label,
                                shape=d[1].shape,
                                dtype=d[1].dtype,
                                data=d[1],
                            )


if HAVE_HDF5:
    HDF5Exporter.register()
