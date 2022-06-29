import pyqtgraph as pg
import pyqtgraph.parametertree as pt

app = pg.mkQApp()

def check(param):
    objs = [pg.mkColor('k'), pg.mkBrush('k'), pg.mkPen('k')]
    results = []
    for obj in objs:
        param.setValue(obj)
        results.append(str(obj) == param.value())
    return results

def test_param():
    param = pt.Parameter.create(name='params', type='str')
    results = check(param)
    assert all(results), results

def test_tree():
    root = pt.Parameter.create(name='params', type='group', children=[
        dict(name='str', type='str')
    ])
    tree = pt.ParameterTree()
    tree.setParameters(root)

    param = root.child('str')
    results = check(param)
    assert all(results), results
