from .. import mkQApp

def test_screenInformation():
    qApp = mkQApp()
    desktop = qApp.desktop()
    resolution = desktop.screenGeometry()
    availableResolution = desktop.availableGeometry()
    print("Screen resolution: {}x{}".format(resolution.width(), resolution.height()))
    print("Available geometry: {}x{}".format(availableResolution.width(), availableResolution.height()))
    print("Number of Screens: {}".format(desktop.screenCount()))
    return None


if __name__ == "__main__":
    test_screenInformation()