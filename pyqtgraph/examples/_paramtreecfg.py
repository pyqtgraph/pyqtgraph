import numpy as np

from pyqtgraph.parametertree.parameterTypes import QtEnumParameter as enum
from pyqtgraph.Qt import QtWidgets

dlg = QtWidgets.QFileDialog

cfg = {
    'list': {
        'limits': {
            'type': 'checklist',
            'limits': ['a', 'b', 'c']
        }
    },
    'file': {
        'acceptMode': {
            'type': 'list',
            'limits': list(enum(dlg.AcceptMode, dlg).enumMap)
        },
        'fileMode': {
            'type': 'list',
            'limits': list(enum(dlg.FileMode, dlg).enumMap)
        },
        'viewMode': {
            'type': 'list',
            'limits': list(enum(dlg.ViewMode, dlg).enumMap)
        },
        'dialogLabel': {
            'type': 'list',
            'limits': list(enum(dlg.DialogLabel, dlg).enumMap)
        },
        'relativeTo': {
            'type': 'str',
            'value': None
        },
        'directory': {
            'type': 'str',
            'value': None
        },
        'windowTitle': {
            'type': 'str',
            'value': None
        },
        'nameFilter': {
            'type': 'str',
            'value': None
        }
    },
    'float': {
        'Float Information': {
            'type': 'str',
            'readonly': True,
            'value': 'Note that all options except "finite" also apply to "int" parameters',
        },
        'step': {
            'type': 'float',
            'limits': [0, None],
            'value': 1,
        },
        'limits': {
            'type': 'list',
            'limits': {'[0, None]': [0, None], '[1, 5]': [1, 5]},
        },
        'suffix': {
            'type': 'list',
            'limits': ['Hz', 's', 'm'],
        },
        'siPrefix': {
            'type': 'bool',
            'value': True
        },
        'finite': {
            'type': 'bool',
            'value': True,
        },
        'dec': {
            'type': 'bool',
            'value': False,
        },
        'minStep': {
            'type': 'float',
            'value': 1.0e-12,
        },
    },

    'checklist': {
        'limits': {
            'type': 'checklist',
            'limits': ['one', 'two', 'three', 'four'],
        },
        'exclusive': {
            'type': 'bool',
            'value': False,
        },
        'delay': {
            'type': 'float',
            'value': 1.0,
            'limits': [0, None]
        }
    },

    'pen': {
        'Pen Information': {
            'type': 'str',
            'value': 'Click the button to see options',
            'readonly': True,
        },
    },

    'slider': {
        'step': {
            'type': 'float',
            'limits': [0, None],
            'value': 1, },
        'format': {
            'type': 'str',
            'value': '{0:>3}',
        },
        'precision': {
            'type': 'int',
            'value': 2,
            'limits': [1, None],
        },
        'span': {
            'type': 'list',
            'limits': {'linspace(-pi, pi)': np.linspace(-np.pi, np.pi), 'arange(10)**2': np.arange(10) ** 2},
        },

        'How to Set': {
            'type': 'list',
            'limits': ['Use span', 'Use step + limits'],
        }
    },

    'action': {
        'shortcut': {
            'type': 'str',
            'value': "Ctrl+Shift+P",
        },
        'icon': {
            'type': 'file',
            'value': None,
            'nameFilter': "Images (*.png *.jpg *.bmp *.jpeg *.svg)",
        },
    },

    'calendar': {
        'format': {
            'type': 'str',
            'value': 'MM DD',
        }
    },

    'Applies to All Types': {
        'Extra Information': {
            'type': 'text',
            'value': 'These apply to all parameters. Watch how this text box is altered by any setting you change.',
            'default': 'These apply to all parameters. Watch how this text box is altered by any setting you change.',
            'readonly': True,
        },
        'readonly': {
            'type': 'bool',
            'value': True,
        },
        'removable': {
            'type': 'bool',
            'tip': 'Adds a context menu option to remove this parameter',
            'value': False,
        },
        'visible': {
            'type': 'bool',
            'value': True,
        },
        'disabled': {
            'type': 'bool',
            'value': False,
        },
        'title': {
            'type': 'str',
            'value': 'Meta Options',
        },
        'default': {
            'tip': 'The default value that gets set when clicking the arrow in the right column',
            'type': 'str',
        },
        'expanded': {
            'type': 'bool',
            'value': True,
        },
    },

    'No Extra Options': {
        'text': 'Unlike the other parameters shown, these don\'t have extra settable options.\n' \
                + 'Note: "int" *does* have the same options as float, mentioned above',
        'int': 10,
        'str': 'Hi, world!',
        'color': '#fff',
        'bool': False,
        'colormap': None,
        'progress': 50,
        'font': 'Inter',
    }
}
