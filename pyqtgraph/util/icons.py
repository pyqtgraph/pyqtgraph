from ..Qt import QtGui, QtWidgets

def iconToQIcon(icon):
    """Convert various icon formats to QIcon.

    Parameters
    ----------
    icon : QIcon, str, int, or None
        The icon to convert. Can be:
        - A QIcon instance (returned as-is)
        - A file path (str) to an icon image
        - A QStyle.StandardPixmap enum value (int)
        - None (returns empty QIcon)

    Returns
    -------
    QIcon
        The converted QIcon instance
    """
    if icon is None:
        return QtGui.QIcon()
    elif isinstance(icon, QtGui.QIcon):
        return icon
    elif isinstance(icon, (int, QtWidgets.QStyle.StandardPixmap)):
        # It's a StandardPixmap enum value - get the standard icon from the application style
        style = QtWidgets.QApplication.instance().style()
        return style.standardIcon(QtWidgets.QStyle.StandardPixmap(icon))
    else:
        # Assume it's a file path or other QIcon-compatible argument
        return QtGui.QIcon(icon)