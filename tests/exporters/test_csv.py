
import csv
import math
import tempfile

import numpy as np
import pytest

import pyqtgraph as pg

app = pg.mkQApp()

@pytest.mark.parametrize('log_mapping', [True, False])
def test_CSVExporter(log_mapping: bool):
    plt = pg.PlotWidget()
    plt.show()
    y1 = [1,3,2,3,1,6,9,8,4,2]
    plt.plot(y=y1, name='myPlot')

    y2 = [3,4,6,1,2,4,2,3,5,3,5,1,3]
    x2 = np.linspace(0, 1.0, len(y2))
    plt.plot(x=x2, y=y2)

    y3 = [1,5,2,3,4,6,1,2,4,2,3,5,3]
    x3 = np.linspace(0, 1.0, len(y3)+1)
    plt.plot(x=x3, y=y3, stepMode="center")

    # log mapping is True tests original data export in log mapped mode
    plt.setLogMode(x=log_mapping, y=log_mapping)
    ex = pg.exporters.CSVExporter(plt.plotItem)
    with tempfile.NamedTemporaryFile(mode="w+t", suffix='.csv', encoding="utf-8", delete=False) as tf:
        print(f"  using {tf.name} as a temporary file")
        ex.export(fileName=tf.name)
        lines = list(csv.reader(tf))
    header = lines.pop(0)
    assert header == ['myPlot_x', 'myPlot_y', 'x0001', 'y0001', 'x0002', 'y0002']

    for i, vals in enumerate(lines):
        vals = list(map(str.strip, vals))
        assert (i >= len(y1) and vals[0] == '') or math.isclose(float(vals[0]), i)
        assert (i >= len(y1) and vals[1] == '') or math.isclose(float(vals[1]), y1[i])

        assert (i >= len(x2) and vals[2] == '') or math.isclose(float(vals[2]), x2[i])
        assert (i >= len(y2) and vals[3] == '') or math.isclose(float(vals[3]), y2[i])

        assert (i >= len(x3) and vals[4] == '') or math.isclose(float(vals[4]), x3[i])
        assert (i >= len(y3) and vals[5] == '') or math.isclose(float(vals[5]), y3[i])


def test_CSVExporter_with_ErrorBarItem():
    plt = pg.PlotWidget()
    plt.show()
    x=np.arange(5)
    y=np.array([1, 2, 3, 2, 1])
    top_error = np.array([2, 3, 3, 3, 2])
    bottom_error = np.array([-2.5, -2.5, -2.5, -2.5, -1.5])

    err = pg.ErrorBarItem(
        x=x,
        y=y,
        top=top_error,
        bottom=bottom_error
    )
    plt.addItem(err)
    ex = pg.exporters.CSVExporter(plt.plotItem)
    with tempfile.NamedTemporaryFile(
        mode="w+t",
        suffix='.csv',
        encoding="utf-8",
        delete=False
    ) as tf:
        ex.export(fileName=tf.name)
        lines = list(csv.reader(tf))

    header = lines.pop(0)

    assert header == ['x0000_error', 'y0000_error', 'y_min_error_0000', 'y_max_error_0000']
    for i, values in enumerate(lines):
        assert pytest.approx(float(values[0])) == x[i]
        assert pytest.approx(float(values[1])) == y[i]
        assert pytest.approx(float(values[2])) == bottom_error[i]
        assert pytest.approx(float(values[3])) == top_error[i]
