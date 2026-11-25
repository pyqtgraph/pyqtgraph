import pyqtgraph as pg
from pyqtgraph.flowchart import Flowchart


app = pg.mkQApp()


def test_flowchart_ctrl_widget_chart_widget_method():
    flowchart = Flowchart()
    ctrl = flowchart.widget()

    chart_widget = ctrl.chartWidget()

    assert chart_widget.scene() is ctrl.scene()
    assert chart_widget.viewBox() is ctrl.viewBox()


def test_bin_op_node_tuple_attribute_error():
    import pytest
    from pyqtgraph.flowchart.library.Operators import DivideNode

    node = DivideNode('divide')
    with pytest.raises(AttributeError):
        node.process(A=object(), B=1)

