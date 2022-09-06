from .. import mkQApp
from ..Qt import QtGui


def test_screenInformation():
    # a qApp is still needed, otherwise screen is None
    qApp = mkQApp()  # noqa
    screen = QtGui.QGuiApplication.primaryScreen()
    screens = QtGui.QGuiApplication.screens()
    resolution = screen.size()
    availableResolution = screen.availableSize()
    print("Screen resolution: {}x{}".format(resolution.width(), resolution.height()))
    print("Available geometry: {}x{}".format(availableResolution.width(), availableResolution.height()))
    print("Number of Screens: {}".format(len(screens)))
    return None


if __name__ == "__main__":
    test_screenInformation()
