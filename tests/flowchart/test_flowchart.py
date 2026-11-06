import pytest

import pyqtgraph as pg
from pyqtgraph.flowchart import Flowchart
from pyqtgraph.flowchart.library.Operators import DivideNode


app = pg.mkQApp()


def test_flowchart_ctrl_widget_chart_widget_method():
    flowchart = Flowchart()
    ctrl = flowchart.widget()

    chart_widget = ctrl.chartWidget()

    assert chart_widget.scene() is ctrl.scene()
    assert chart_widget.viewBox() is ctrl.viewBox()


def test_binop_node_without_operator_raises_attribute_error():
    node = DivideNode('Divide')
    assert node.process(A=6.0, B=3.0, display=False)['Out'] == 2.0
    with pytest.raises(AttributeError):
        node.process(A=object(), B=3.0, display=False)
