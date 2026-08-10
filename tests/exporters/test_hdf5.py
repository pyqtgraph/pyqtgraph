import numpy as np
import pytest
from numpy.testing import assert_equal

import pyqtgraph as pg
from pyqtgraph.exporters import HDF5Exporter
from pyqtgraph.Qt import QtCore


h5py = pytest.importorskip("h5py")


translate = QtCore.QCoreApplication.translate


@pytest.fixture
def tmp_h5(tmp_path):
    yield tmp_path / "data.h5"


@pytest.mark.parametrize("combine", [False, True])
def test_HDF5Exporter(tmp_h5, combine):
    # Test export with multiple curves of different size.
    # Tests both options for stacking the data (columnMode).
    x1 = np.linspace(0, 1, 10)
    y1 = np.sin(x1)
    x2 = np.linspace(0, 1, 100)
    y2 = np.cos(x2)

    plt = pg.PlotWidget()
    plt.show()
    plt.plot(x=x1, y=y1)
    plt.plot(x=x2, y=y2)

    ex = HDF5Exporter(plt.plotItem)

    if not combine:
        ex.parameters()['columnMode'] = translate("Exporter", "separate x and y")

    ex.export(fileName=tmp_h5)

    with h5py.File(tmp_h5, 'r') as f:
        if combine:
            # should be two datasets with default names
            dset1 = f["Curve 1"]
            assert isinstance(dset1, h5py.Dataset)
            dset2 = f["Curve 2"]
            assert isinstance(dset2, h5py.Dataset)
            assert_equal(np.column_stack((x1, y1)), dset1)
            assert_equal(np.column_stack((x2, y2)), dset2)
        else:
            # should be two groups with default names,
            # two datasets with default names in each
            assert isinstance(f["Curve 1"], h5py.Group)
            assert_equal(x1, f["Curve 1"]["x"])
            assert_equal(y1, f["Curve 1"]["y"])
            assert isinstance(f["Curve 2"], h5py.Group)
            assert_equal(x2, f["Curve 2"]["x"])
            assert_equal(y2, f["Curve 2"]["y"])


@pytest.mark.parametrize("combine", [False, True])
def test_HDF5Exporter_2plots(tmp_h5, combine):
    # Test export with multiple curves of different size on two plots.
    # Tests both options for stacking the data (columnMode).
    x1 = np.linspace(0, 1, 10)
    y1 = np.sin(x1)
    x2 = np.linspace(0, 1, 100)
    y2 = np.cos(x2)

    plt: pg.GraphicsLayoutWidget = pg.GraphicsLayoutWidget()
    canvas1: pg.PlotItem = plt.ci.addPlot(row=0, col=0)
    canvas2: pg.PlotItem = plt.ci.addPlot(row=1, col=0, title="Second plot")
    plt.show()
    canvas1.plot(x=x1, y=y1)
    canvas1.plot(x=x2, y=y2)
    canvas2.plot(x=y1, y=x1)
    canvas2.plot(x=y2, y=x2)

    ex = HDF5Exporter(plt.sceneObj)

    if not combine:
        ex.parameters()['columnMode'] = translate("Exporter", "separate x and y")

    ex.export(fileName=tmp_h5)

    with h5py.File(tmp_h5, 'r') as f:
        if combine:
            # should be two datasets with default names
            group1 = f["Plot 1"]
            assert isinstance(group1, h5py.Group)
            dset1 = group1["Curve 1"]
            assert isinstance(dset1, h5py.Dataset)
            dset2 = group1["Curve 2"]
            assert isinstance(dset2, h5py.Dataset)
            assert_equal(np.column_stack((x1, y1)), dset1)
            assert_equal(np.column_stack((x2, y2)), dset2)
            group1 = f["Second plot"]
            assert isinstance(group1, h5py.Group)
            dset1 = group1["Curve 1"]
            assert isinstance(dset1, h5py.Dataset)
            dset2 = group1["Curve 2"]
            assert isinstance(dset2, h5py.Dataset)
            assert_equal(np.column_stack((y1, x1)), dset1)
            assert_equal(np.column_stack((y2, x2)), dset2)
        else:
            # should be two groups with default names,
            # two datasets with default names in each
            group1 = f["Plot 1"]
            assert isinstance(group1, h5py.Group)
            assert isinstance(group1["Curve 1"], h5py.Group)
            assert_equal(x1, group1["Curve 1"]["x"])
            assert_equal(y1, group1["Curve 1"]["y"])
            assert_equal(x2, group1["Curve 2"]["x"])
            assert_equal(y2, group1["Curve 2"]["y"])
            group2 = f["Second plot"]
            assert isinstance(group2, h5py.Group)
            assert isinstance(group2["Curve 1"], h5py.Group)
            assert_equal(x1, group2["Curve 1"]["y"])
            assert_equal(y1, group2["Curve 1"]["x"])
            assert_equal(x2, group2["Curve 2"]["y"])
            assert_equal(y2, group2["Curve 2"]["x"])
