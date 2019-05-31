from .. import mkQApp

qApp = mkQApp()


def test_displayResolution():
    desktop = qApp.desktop().screenGeometry()
    width, height = desktop.width(), desktop.height()
    print("\n\nDisplay Resolution Logged as {}x{}\n\n".format(width, height))
    assert height > 0 and width > 0
