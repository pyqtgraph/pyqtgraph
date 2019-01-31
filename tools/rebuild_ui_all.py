import os
import subprocess

PATH = os.path.dirname(__file__)


TEMPLATE_FILES = [
    '../pyqtgraph/canvas/CanvasTemplate.ui',
    '../pyqtgraph/canvas/TransformGuiTemplate.ui',
    '../pyqtgraph/console/template.ui',
    '../pyqtgraph/flowchart/FlowchartCtrlTemplate.ui',
    '../pyqtgraph/flowchart/FlowchartTemplate.ui',
    '../pyqtgraph/graphicsItems/PlotItem/plotConfigTemplate.ui',
    '../pyqtgraph/graphicsItems/ViewBox/axisCtrlTemplate.ui',
    '../pyqtgraph/GraphicsScene/exportDialogTemplate.ui',
    '../pyqtgraph/imageview/ImageViewTemplate.ui',
    '../pyqtgraph/tests/uictest.ui',
    '../pyqtgraph/tests/uictest.ui',

    '../examples/designerExample.ui',
    '../examples/exampleLoaderTemplate.ui',
    '../examples/ScatterPlotSpeedTestTemplate.ui',
    '../examples/VideoTemplate.ui',
    ]


if __name__ == '__main__':

    procs = []
    for file in TEMPLATE_FILES:
        rebuild = os.path.join(PATH, 'rebuildUi.py')
        # procs.append(subprocess.Popen(['python', rebuild, '--force', os.path.join(PATH, file)]))
        procs.append(subprocess.Popen(['python', rebuild, os.path.join(PATH, file)]))

    while True:
        for i in reversed(range(len(procs))):
            proc = procs[i]
            if proc.poll() is not None:
                procs.pop(i)

        if len(procs) == 0:
            break

    print('All processes finished!')
