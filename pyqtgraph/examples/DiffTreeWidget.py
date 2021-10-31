
"""
Simple use of DiffTreeWidget to display differences between structures of 
nested dicts, lists, and arrays.
"""

import numpy as np

import pyqtgraph as pg

app = pg.mkQApp("DiffTreeWidget Example")
A = {
    'a list': [1,2,2,4,5,6, {'nested1': 'aaaa', 'nested2': 'bbbbb'}, "seven"],
    'a dict': {
        'x': 1,
        'y': 2,
        'z': 'three'
    },
    'an array': np.random.randint(10, size=(40,10)),
    #'a traceback': some_func1(),
    #'a function': some_func1,
    #'a class': pg.DataTreeWidget,
}

B = {
    'a list': [1,2,3,4,5,5, {'nested1': 'aaaaa', 'nested2': 'bbbbb'}, "seven"],
    'a dict': {
        'x': 2,
        'y': 2,
        'z': 'three',
        'w': 5
    },
    'another dict': {1:2, 2:3, 3:4},
    'an array': np.random.randint(10, size=(40,10)),
}

tree = pg.DiffTreeWidget()
tree.setData(A, B)
tree.show()
tree.setWindowTitle('pyqtgraph example: DiffTreeWidget')
tree.resize(1000, 800)


if __name__ == '__main__':
    pg.exec()
