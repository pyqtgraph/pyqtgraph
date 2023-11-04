from typing import Union
from typing_extensions import Unpack

from ..Node import Node
from .common import CtrlNode


class UniOpNode(Node):
    """Generic node for performing any operation like Out = In.fn()"""

    def __init__(self, name: str, fn: str) -> None:
        self.fn = fn
        Node.__init__(self, name, terminals={
            'In': {'io': 'in'},
            'Out': {'io': 'out', 'bypass': 'In'}
        })

    def process(self, **args: Unpack) -> dict:
        return {'Out': getattr(args['In'], self.fn)()}


class BinOpNode(CtrlNode):
    """Generic node for performing any operation like A.fn(B)"""

    _dtypes: list[str] = [
        'float64', 'float32', 'float16',
        'int64', 'int32', 'int16', 'int8',
        'uint64', 'uint32', 'uint16', 'uint8'
    ]

    uiTemplate: list[tuple] = [
        ('outputType', 'combo', {'values': ['no change', 'input A', 'input B'] + _dtypes, 'index': 0})
    ]

    def __init__(self, name: str, fn: Union[tuple[str, ...], str]) -> None:
        self.fn: Union[tuple[str, ...], str] = fn
        CtrlNode.__init__(self, name, terminals={
            'A': {'io': 'in'},
            'B': {'io': 'in'},
            'Out': {'io': 'out', 'bypass': 'A'}
        })

    def process(self, **args: Unpack) -> dict:  # type: ignore
        if isinstance(self.fn, tuple):
            for name in self.fn:
                try:
                    fn = getattr(args['A'], name)
                    break
                except AttributeError as e:
                    pass
            else:
                raise AttributeError()
        else:
            fn = getattr(args['A'], self.fn)
        out = fn(args['B'])
        if out is NotImplemented:
            raise Exception(
                "Operation %s not implemented between %s and %s" % (fn, str(type(args['A'])), str(type(args['B']))))

        # Coerce dtype if requested
        typ = self.stateGroup.state()['outputType']
        if typ == 'no change':
            pass
        elif typ == 'input A':
            out = out.astype(args['A'].dtype)
        elif typ == 'input B':
            out = out.astype(args['B'].dtype)
        else:
            out = out.astype(typ)

        # print "     ", fn, out
        return {'Out': out}


class AbsNode(UniOpNode):
    """Returns abs(Inp). Does not check input types."""
    nodeName: str = 'Abs'

    def __init__(self, name: str) -> None:
        UniOpNode.__init__(self, name, '__abs__')


class AddNode(BinOpNode):
    """Returns A + B. Does not check input types."""
    nodeName: str = 'Add'

    def __init__(self, name: str) -> None:
        BinOpNode.__init__(self, name, '__add__')


class SubtractNode(BinOpNode):
    """Returns A - B. Does not check input types."""
    nodeName: str = 'Subtract'

    def __init__(self, name: str) -> None:
        BinOpNode.__init__(self, name, '__sub__')


class MultiplyNode(BinOpNode):
    """Returns A * B. Does not check input types."""
    nodeName: str = 'Multiply'

    def __init__(self, name: str) -> None:
        BinOpNode.__init__(self, name, '__mul__')


class DivideNode(BinOpNode):
    """Returns A / B. Does not check input types."""
    nodeName: str = 'Divide'

    def __init__(self, name: str) -> None:
        # try truediv first, followed by div
        BinOpNode.__init__(self, name, ('__truediv__', '__div__'))


class FloorDivideNode(BinOpNode):
    """Returns A // B. Does not check input types."""
    nodeName: str = 'FloorDivide'

    def __init__(self, name: str) -> None:
        BinOpNode.__init__(self, name, '__floordiv__')
