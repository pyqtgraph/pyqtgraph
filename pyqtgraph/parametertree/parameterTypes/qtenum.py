from enum import Enum
from ...Qt import QT_LIB, QtCore
from .list import ListParameter


class QtEnumParameter(ListParameter):
    def __init__(self, enum, searchObj=QtCore.Qt, **opts):
        """
        Constructs a list of allowed enum values from the enum class provided
        `searchObj` is only needed for PyQt5 compatibility, where it must be the module holding the enum.
        For instance, if making a QtEnumParameter out of QtWidgets.QFileDialog.Option, `searchObj` would
        be QtWidgets.QFileDialog
        """
        self.enum = enum
        self.searchObj = searchObj
        opts.setdefault('name', enum.__name__)
        self.enumMap = self._getAllowedEnums(enum)

        opts.update(limits=self.formattedLimits())
        super().__init__(**opts)

    def setValue(self, value, blockSignal=None):
        if isinstance(value, str):
            value = self.enumMap[value]
        super().setValue(value, blockSignal)

    def formattedLimits(self):
        # Title-cased words without the ending substring for brevity
        substringEnd = None
        mapping = self.enumMap
        shortestName = min(len(name) for name in mapping)
        names = list(mapping)
        cmpName, *names = names
        for ii in range(-1, -shortestName-1, -1):
            if any(cmpName[ii] != curName[ii] for curName in names):
                substringEnd = ii+1
                break
        # Special case of 0: Set to none to avoid null string
        if substringEnd == 0:
            substringEnd = None
        limits = {}
        for kk, vv in self.enumMap.items():
            limits[kk[:substringEnd]] = vv
        return limits

    def saveState(self, filter=None):
        state = super().saveState(filter)
        reverseMap = dict(zip(self.enumMap.values(), self.enumMap))
        state['value'] = reverseMap[state['value']]
        return state

    def _getAllowedEnums(self, enum):
        """Pyside provides a dict for easy evaluation"""
        if issubclass(enum, Enum):
            # PyQt6 and PySide6 (opt-in in 6.3.1) use python enums
            vals = {e.name: e for e in enum}
        elif 'PySide' in QT_LIB:
            vals = enum.values
        elif 'PyQt5' in QT_LIB:
            vals = {}
            for key in dir(self.searchObj):
                value = getattr(self.searchObj, key)
                if isinstance(value, enum):
                    vals[key] = value
        else:
            raise RuntimeError(f'Cannot find associated enum values for qt lib {QT_LIB}')
        # Remove "M<enum>" since it's not a real option
        vals.pop(f'M{enum.__name__}', None)
        return vals
