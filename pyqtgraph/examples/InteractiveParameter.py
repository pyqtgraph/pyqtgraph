from functools import wraps

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.parametertree import Parameter, ParameterTree, RunOpts, InteractiveFunction, interact

app = pg.mkQApp()

class LAST_RESULT:
  """Just for testing purposes"""
  value = None

def printResult(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    LAST_RESULT.value = func(*args, **kwargs)
    QtWidgets.QMessageBox.information(QtWidgets.QApplication.activeWindow(),
                                      'Function Run!', f'Func result: {LAST_RESULT.value}')
  return wrapper

host = Parameter.create(name='Interactive Parameter Use', type='group')

@host.interactDecorator()
@printResult
def easySample(a=5, b=6):
    return a + b

@host.interactDecorator()
@printResult
def hasTooltipInfo(a=4, b=6):
    """
    [a.options]
    tip=I'm the 'A' parameter
    [b.options]
    tip=My limits are from 0 to 10 incrementing by 2
    limits=[0, 10]
    step=2
    """
    return a + b

@host.interactDecorator()
@printResult
def stringParams(a='5', b='6'):
    return a + b

@host.interactDecorator(a=10)
@printResult
def requiredParam(a, b=10):
    return a + b

@host.interactDecorator(ignores=['a'])
@printResult
def ignoredAParam(a=10, b=20):
    return a*b

@host.interactDecorator(runOpts=RunOpts.ON_BUTTON)
@printResult
def runOnButton(a=10, b=20):
    return a + b

x = 5
@printResult
def accessVarInDifferentScope(x, y=10):
    return x + y
func_interactive = InteractiveFunction(accessVarInDifferentScope, deferred={'x': lambda: x})
# Value is redeclared, but still bound
x = 10
interact(func_interactive, parent=host)


with RunOpts.optsContext(title=str.upper):
    @host.interactDecorator()
    @printResult
    def capslocknames(a=5):
        return a

@host.interactDecorator(runOpts=(RunOpts.ON_CHANGED, RunOpts.ON_BUTTON),
                         a={'type': 'list', 'limits': [5, 10, 20]}
                         )
@printResult
def runOnBtnOrChange_listOpts(a=5):
    return a


@host.interactDecorator(nest=False)
@printResult
def onlyTheArgumentsAppear(thisIsAFunctionArg=True):
    return thisIsAFunctionArg

tree = ParameterTree()
tree.setParameters(host)

tree.show()
if __name__ == '__main__':
    pg.exec()
