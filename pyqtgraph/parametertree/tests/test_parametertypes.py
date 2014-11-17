import pyqtgraph.parametertree as pt
import pyqtgraph as pg
app = pg.mkQApp()

def test_opts():
    paramSpec = [
        dict(name='bool', type='bool', readonly=True),
        dict(name='color', type='color', readonly=True),
    ]

    param = pt.Parameter.create(name='params', type='group', children=paramSpec)
    tree = pt.ParameterTree()
    tree.setParameters(param)

    assert param.param('bool').items.keys()[0].widget.isEnabled() is False
    assert param.param('color').items.keys()[0].widget.isEnabled() is False


