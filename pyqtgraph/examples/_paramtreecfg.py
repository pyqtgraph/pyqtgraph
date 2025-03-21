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
            'default': None
        },
        'directory': {
            'type': 'str',
            'default': None
        },
        'windowTitle': {
            'type': 'str',
            'default': None
        },
        'nameFilter': {
            'type': 'str',
            'default': None
        }
    },
    'float': {
        'Float Information': {
            'type': 'str',
            'readonly': True,
            'default': 'Note that all options except "finite" also apply to "int" parameters',
        },
        'step': {
            'type': 'float',
            'limits': [0, None],
            'default': 1,
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
            'default': True
        },
        'finite': {
            'type': 'bool',
            'default': True,
        },
        'dec': {
            'type': 'bool',
            'default': False,
        },
        'minStep': {
            'type': 'float',
            'default': 1.0e-12,
        },
    },

    'checklist': {
        'limits': {
            'type': 'checklist',
            'limits': ['one', 'two', 'three', 'four'],
        },
        'exclusive': {
            'type': 'bool',
            'default': False,
        },
        'delay': {
            'type': 'float',
            'default': 1.0,
            'limits': [0, None]
        }
    },

    'pen': {
        'Pen Information': {
            'type': 'str',
            'default': "",
            'value': 'Click the button to see options',
            'readonly': True,
        },
    },

    'slider': {
        'step': {
            'type': 'float',
            'limits': [0, None],
            'default': 1, },
        'format': {
            'type': 'str',
            'default': '{0:>3}',
        },
        'precision': {
            'type': 'int',
            'default': 2,
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
            'default': "Ctrl+Shift+P",
        },
        'icon': {
            'type': 'file',
            'default': None,
            'nameFilter': "Images (*.png *.jpg *.bmp *.jpeg *.svg)",
        },
    },

    'calendar': {
        'format': {
            'type': 'str',
            'default': 'MM DD',
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
            'default': True,
        },
        'removable': {
            'type': 'bool',
            'tip': 'Adds a context menu option to remove this parameter',
            'default': False,
        },
        'visible': {
            'type': 'bool',
            'default': True,
        },
        'disabled': {
            'type': 'bool',
            'default': False,
        },
        'title': {
            'type': 'str',
            'default': 'Meta Options',
        },
        'default': {
            'tip': 'The default value that gets set when clicking the arrow in the right column',
            'type': 'str',
        },
        'expanded': {
            'type': 'bool',
            'default': True,
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
        'cmaplut' : 'viridis',
        'progress': 50,
        'font': 'Inter',
    }
}
