from .. import mkQApp


def getResolution():
    qApp = mkQApp()
    desktop = qApp.desktop().screenGeometry()
    return (desktop.width(), desktop.height())
