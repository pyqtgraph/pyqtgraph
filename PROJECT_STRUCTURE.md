# Project Structure: `pyqtgraph`

- **Root Path**: `D:\Work\Opensource\pyqtgraph`
- **Generated On**: 2026-11-06 13:21:43
- **Total Folders**: 74
- **Total Files**: 793

```text
pyqtgraph/
├── benchmarks/
│   ├── __init__.py
│   ├── arrayToFillPaths.py
│   ├── arrayToQPath.py
│   └── renderImageItem.py
├── binder/
│   ├── apt.txt
│   ├── requirements.txt
│   ├── runtime.txt
│   └── start
├── doc/
│   ├── source/
│   │   ├── _static/
│   │   │   ├── custom.css
│   │   │   ├── dash.css
│   │   │   ├── docset-icon.png
│   │   │   ├── docset-icon@2x.png
│   │   │   ├── peegee_03_square_no_bg_32_cleaned.ico
│   │   │   └── peegee_04_square_no_bg_180_cleaned.png
│   │   ├── api_reference/
│   │   │   ├── 3dgraphics/
│   │   │   │   ├── glaxisitem.rst
│   │   │   │   ├── glgraphicsitem.rst
│   │   │   │   ├── glgraphitem.rst
│   │   │   │   ├── glgriditem.rst
│   │   │   │   ├── glimageitem.rst
│   │   │   │   ├── gllineplotitem.rst
│   │   │   │   ├── glmeshitem.rst
│   │   │   │   ├── glscatterplotitem.rst
│   │   │   │   ├── glsurfaceplotitem.rst
│   │   │   │   ├── gltextitem.rst
│   │   │   │   ├── glviewwidget.rst
│   │   │   │   ├── glvolumeitem.rst
│   │   │   │   ├── index.rst
│   │   │   │   └── meshdata.rst
│   │   │   ├── exporters/
│   │   │   │   ├── Exporter.rst
│   │   │   │   ├── index.rst
│   │   │   │   └── SVGExporter.rst
│   │   │   ├── flowchart/
│   │   │   │   ├── flowchart.rst
│   │   │   │   ├── index.rst
│   │   │   │   ├── node.rst
│   │   │   │   └── terminal.rst
│   │   │   ├── graphicsItems/
│   │   │   │   ├── arrowitem.rst
│   │   │   │   ├── axisitem.rst
│   │   │   │   ├── bargraphitem.rst
│   │   │   │   ├── buttonitem.rst
│   │   │   │   ├── colorbaritem.rst
│   │   │   │   ├── curvearrow.rst
│   │   │   │   ├── curvepoint.rst
│   │   │   │   ├── dateaxisitem.rst
│   │   │   │   ├── errorbaritem.rst
│   │   │   │   ├── fillbetweenitem.rst
│   │   │   │   ├── gradienteditoritem.rst
│   │   │   │   ├── gradientlegend.rst
│   │   │   │   ├── graphicsitem.rst
│   │   │   │   ├── graphicslayout.rst
│   │   │   │   ├── graphicsobject.rst
│   │   │   │   ├── graphicswidget.rst
│   │   │   │   ├── graphicswidgetanchor.rst
│   │   │   │   ├── graphitem.rst
│   │   │   │   ├── griditem.rst
│   │   │   │   ├── histogramlutitem.rst
│   │   │   │   ├── imageitem.rst
│   │   │   │   ├── index.rst
│   │   │   │   ├── infiniteline.rst
│   │   │   │   ├── isocurveitem.rst
│   │   │   │   ├── labelitem.rst
│   │   │   │   ├── legenditem.rst
│   │   │   │   ├── linearregionitem.rst
│   │   │   │   ├── pcolormeshitem.rst
│   │   │   │   ├── plotcurveitem.rst
│   │   │   │   ├── plotdataitem.rst
│   │   │   │   ├── plotitem.rst
│   │   │   │   ├── roi.rst
│   │   │   │   ├── scalebar.rst
│   │   │   │   ├── scatterplotitem.rst
│   │   │   │   ├── targetitem.rst
│   │   │   │   ├── textitem.rst
│   │   │   │   ├── uigraphicsitem.rst
│   │   │   │   ├── viewbox.rst
│   │   │   │   └── vtickgroup.rst
│   │   │   ├── graphicsscene/
│   │   │   │   ├── graphicsscene.rst
│   │   │   │   ├── hoverevent.rst
│   │   │   │   ├── index.rst
│   │   │   │   ├── mouseclickevent.rst
│   │   │   │   └── mousedragevent.rst
│   │   │   ├── parametertree/
│   │   │   │   ├── apiref.rst
│   │   │   │   ├── index.rst
│   │   │   │   ├── interactiveparameters.rst
│   │   │   │   ├── parameter.rst
│   │   │   │   ├── parameteritem.rst
│   │   │   │   ├── parametertree.rst
│   │   │   │   └── parametertypes.rst
│   │   │   ├── widgets/
│   │   │   │   ├── busycursor.rst
│   │   │   │   ├── checktable.rst
│   │   │   │   ├── colorbutton.rst
│   │   │   │   ├── colormapmenu.rst
│   │   │   │   ├── colormapwidget.rst
│   │   │   │   ├── combobox.rst
│   │   │   │   ├── consolewidget.rst
│   │   │   │   ├── datatreewidget.rst
│   │   │   │   ├── feedbackbutton.rst
│   │   │   │   ├── filedialog.rst
│   │   │   │   ├── gradientwidget.rst
│   │   │   │   ├── graphicslayoutwidget.rst
│   │   │   │   ├── graphicsview.rst
│   │   │   │   ├── histogramlutwidget.rst
│   │   │   │   ├── imageview.rst
│   │   │   │   ├── index.rst
│   │   │   │   ├── joystickbutton.rst
│   │   │   │   ├── layoutwidget.rst
│   │   │   │   ├── matplotlibwidget.rst
│   │   │   │   ├── pathbutton.rst
│   │   │   │   ├── plotwidget.rst
│   │   │   │   ├── progressdialog.rst
│   │   │   │   ├── rawimagewidget.rst
│   │   │   │   ├── remotegraphicsview.rst
│   │   │   │   ├── scatterplotwidget.rst
│   │   │   │   ├── spinbox.rst
│   │   │   │   ├── tablewidget.rst
│   │   │   │   ├── treewidget.rst
│   │   │   │   ├── valuelabel.rst
│   │   │   │   └── verticallabel.rst
│   │   │   ├── colormap.rst
│   │   │   ├── config_options.rst
│   │   │   ├── dockarea.rst
│   │   │   ├── functions.rst
│   │   │   ├── index.rst
│   │   │   ├── point.rst
│   │   │   ├── transform3d.rst
│   │   │   └── uml_overview.rst
│   │   ├── developer_guide/
│   │   │   ├── index.rst
│   │   │   └── internals.rst
│   │   ├── dictionaries/
│   │   │   ├── custom.dic
│   │   │   ├── numpy.dic
│   │   │   ├── PyQt6.QtCore.dic
│   │   │   ├── PyQt6.QtGui.dic
│   │   │   ├── PyQt6.QtWidgets.dic
│   │   │   ├── pyqtgraph.dic
│   │   │   └── sphinx.ext.autodoc.dic
│   │   ├── getting_started/
│   │   │   ├── 3dgraphics.rst
│   │   │   ├── how_to_use.rst
│   │   │   ├── images.rst
│   │   │   ├── index.rst
│   │   │   ├── installation.rst
│   │   │   ├── introduction.rst
│   │   │   ├── plotting.rst
│   │   │   ├── prototyping.rst
│   │   │   └── qtcrashcourse.rst
│   │   ├── images/
│   │   │   ├── example_false_color_image.png
│   │   │   ├── example_gradient_plot.png
│   │   │   ├── example_imageitem_transform.png
│   │   │   ├── gen_example_false_color_image.py
│   │   │   ├── gen_example_gradient_plot.py
│   │   │   ├── gen_example_imageitem_transform.py
│   │   │   ├── index_api.svg
│   │   │   ├── index_contribute.svg
│   │   │   ├── index_getting_started.svg
│   │   │   ├── index_user_guide.svg
│   │   │   ├── overview_uml-dark_mode.graphml
│   │   │   ├── overview_uml-dark_mode.svg
│   │   │   ├── overview_uml-light_mode.graphml
│   │   │   ├── overview_uml-light_mode.svg
│   │   │   ├── peegee_02.svg
│   │   │   ├── plottingClasses.png
│   │   │   └── plottingClasses.svg
│   │   ├── user_guide/
│   │   │   ├── exporting.rst
│   │   │   ├── index.rst
│   │   │   ├── mouse_interaction.rst
│   │   │   ├── region_of_interest.rst
│   │   │   └── style.rst
│   │   ├── conf.py
│   │   └── index.rst
│   ├── listmissing.py
│   ├── make.bat
│   ├── Makefile
│   ├── requirements.txt
│   └── uml_overview.txt
├── pyqtgraph/
│   ├── colors/
│   │   ├── maps/
│   │   │   ├── CC-BY license - applies to CET color map data.txt
│   │   │   ├── CC0 legal code - applies to virids, magma, plasma, inferno and cividis.txt
│   │   │   ├── CET-C1.csv
│   │   │   ├── CET-C1s.csv
│   │   │   ├── CET-C2.csv
│   │   │   ├── CET-C2s.csv
│   │   │   ├── CET-C3.csv
│   │   │   ├── CET-C3s.csv
│   │   │   ├── CET-C4.csv
│   │   │   ├── CET-C4s.csv
│   │   │   ├── CET-C5.csv
│   │   │   ├── CET-C5s.csv
│   │   │   ├── CET-C6.csv
│   │   │   ├── CET-C6s.csv
│   │   │   ├── CET-C7.csv
│   │   │   ├── CET-C7s.csv
│   │   │   ├── CET-CBC1.csv
│   │   │   ├── CET-CBC2.csv
│   │   │   ├── CET-CBD1.csv
│   │   │   ├── CET-CBL1.csv
│   │   │   ├── CET-CBL2.csv
│   │   │   ├── CET-CBTC1.csv
│   │   │   ├── CET-CBTC2.csv
│   │   │   ├── CET-CBTD1.csv
│   │   │   ├── CET-CBTL1.csv
│   │   │   ├── CET-CBTL2.csv
│   │   │   ├── CET-D1.csv
│   │   │   ├── CET-D10.csv
│   │   │   ├── CET-D11.csv
│   │   │   ├── CET-D12.csv
│   │   │   ├── CET-D13.csv
│   │   │   ├── CET-D1A.csv
│   │   │   ├── CET-D2.csv
│   │   │   ├── CET-D3.csv
│   │   │   ├── CET-D4.csv
│   │   │   ├── CET-D6.csv
│   │   │   ├── CET-D7.csv
│   │   │   ├── CET-D8.csv
│   │   │   ├── CET-D9.csv
│   │   │   ├── CET-I1.csv
│   │   │   ├── CET-I2.csv
│   │   │   ├── CET-I3.csv
│   │   │   ├── CET-L1.csv
│   │   │   ├── CET-L10.csv
│   │   │   ├── CET-L11.csv
│   │   │   ├── CET-L12.csv
│   │   │   ├── CET-L13.csv
│   │   │   ├── CET-L14.csv
│   │   │   ├── CET-L15.csv
│   │   │   ├── CET-L16.csv
│   │   │   ├── CET-L17.csv
│   │   │   ├── CET-L18.csv
│   │   │   ├── CET-L19.csv
│   │   │   ├── CET-L2.csv
│   │   │   ├── CET-L3.csv
│   │   │   ├── CET-L4.csv
│   │   │   ├── CET-L5.csv
│   │   │   ├── CET-L6.csv
│   │   │   ├── CET-L7.csv
│   │   │   ├── CET-L8.csv
│   │   │   ├── CET-L9.csv
│   │   │   ├── CET-R1.csv
│   │   │   ├── CET-R2.csv
│   │   │   ├── CET-R3.csv
│   │   │   ├── CET-R4.csv
│   │   │   ├── cividis.csv
│   │   │   ├── inferno.csv
│   │   │   ├── magma.csv
│   │   │   ├── PAL-relaxed.hex
│   │   │   ├── PAL-relaxed_bright.hex
│   │   │   ├── plasma.csv
│   │   │   ├── turbo.csv
│   │   │   └── viridis.csv
│   │   ├── __init__.py
│   │   └── palette.py
│   ├── console/
│   │   ├── __init__.py
│   │   ├── CmdInput.py
│   │   ├── Console.py
│   │   ├── exception_widget.py
│   │   ├── repl_widget.py
│   │   └── stackwidget.py
│   ├── dockarea/
│   │   ├── __init__.py
│   │   ├── Container.py
│   │   ├── Dock.py
│   │   ├── DockArea.py
│   │   └── DockDrop.py
│   ├── examples/
│   │   ├── notebooks/
│   │   │   ├── ImageAnalysis.ipynb
│   │   │   ├── PColorMeshItem.ipynb
│   │   │   ├── Plotting.ipynb
│   │   │   ├── ROIExamples.ipynb
│   │   │   ├── ScatterPlot.ipynb
│   │   │   └── simple.ipynb
│   │   ├── optics/
│   │   │   ├── __init__.py
│   │   │   ├── pyoptic.py
│   │   │   └── schott_glasses.csv.gz
│   │   ├── relativity/
│   │   │   ├── presets/
│   │   │   │   ├── Grid Expansion.cfg
│   │   │   │   ├── Twin Paradox (grid).cfg
│   │   │   │   └── Twin Paradox.cfg
│   │   │   ├── __init__.py
│   │   │   └── relativity.py
│   │   ├── verlet_chain/
│   │   │   ├── __init__.py
│   │   │   ├── chain.py
│   │   │   ├── make
│   │   │   ├── relax.c
│   │   │   └── relax.py
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── _buildParamTypes.py
│   │   ├── _paramtreecfg.py
│   │   ├── Arrow.py
│   │   ├── AxisItem_label_overlap.py
│   │   ├── BarGraphItem.py
│   │   ├── beeswarm.py
│   │   ├── BoxplotItem.py
│   │   ├── CLIexample.py
│   │   ├── ColorBarItem.py
│   │   ├── ColorButton.py
│   │   ├── ColorGradientPlots.py
│   │   ├── colorMaps.py
│   │   ├── colorMapsLinearized.py
│   │   ├── console_exception_inspection.py
│   │   ├── ConsoleWidget.py
│   │   ├── contextMenu.py
│   │   ├── crosshair.py
│   │   ├── customGraphicsItem.py
│   │   ├── CustomGraphItem.py
│   │   ├── customPlot.py
│   │   ├── DataSlicing.py
│   │   ├── DataTreeWidget.py
│   │   ├── DateAxisItem.py
│   │   ├── DiffTreeWidget.py
│   │   ├── dockarea.py
│   │   ├── Draw.py
│   │   ├── ErrorBarItem.py
│   │   ├── ExampleApp.py
│   │   ├── exampleLoaderTemplate.py
│   │   ├── exampleLoaderTemplate_generic.py
│   │   ├── FillBetweenItem.py
│   │   ├── Flowchart.py
│   │   ├── FlowchartCustomNode.py
│   │   ├── fractal.py
│   │   ├── GLBarGraphItem.py
│   │   ├── GLGradientLegendItem.py
│   │   ├── GLGraphItem.py
│   │   ├── GLGridAxisItem.py
│   │   ├── GLImageItem.py
│   │   ├── GLIsosurface.py
│   │   ├── GLLinePlotItem.py
│   │   ├── GLMeshItem.py
│   │   ├── glow.py
│   │   ├── GLPainterItem.py
│   │   ├── GLScatterPlotItem.py
│   │   ├── GLshaders.py
│   │   ├── GLSurfacePlot.py
│   │   ├── GLTextItem.py
│   │   ├── GLViewWidget.py
│   │   ├── GLVolumeItem.py
│   │   ├── GradientEditor.py
│   │   ├── GradientWidget.py
│   │   ├── GraphicsLayout.py
│   │   ├── GraphicsScene.py
│   │   ├── GraphItem.py
│   │   ├── hdf5.py
│   │   ├── histogram.py
│   │   ├── HistogramLUT.py
│   │   ├── imageAnalysis.py
│   │   ├── ImageItem.py
│   │   ├── ImageView.py
│   │   ├── InfiniteLine.py
│   │   ├── infiniteline_performance.py
│   │   ├── InteractiveParameter.py
│   │   ├── isocurve.py
│   │   ├── JoystickButton.py
│   │   ├── jupyter_console_example.py
│   │   ├── Legend.py
│   │   ├── linkedViews.py
│   │   ├── logAxis.py
│   │   ├── LogPlotTest.py
│   │   ├── MatrixDisplayExample.py
│   │   ├── MouseSelection.py
│   │   ├── MultiDataPlot.py
│   │   ├── MultiplePlotAxes.py
│   │   ├── multiplePlotSpeedTest.py
│   │   ├── MultiPlotSpeedTest.py
│   │   ├── multiprocess.py
│   │   ├── NonUniformImage.py
│   │   ├── optics_demos.py
│   │   ├── PanningPlot.py
│   │   ├── parallelize.py
│   │   ├── parametertree.py
│   │   ├── PColorMeshItem.py
│   │   ├── PlotAutoRange.py
│   │   ├── PlotSpeedTest.py
│   │   ├── Plotting.py
│   │   ├── PlotWidget.py
│   │   ├── ProgressDialog.py
│   │   ├── relativity_demo.py
│   │   ├── RemoteGraphicsView.py
│   │   ├── RemoteSpeedTest.py
│   │   ├── ROIExamples.py
│   │   ├── ROItypes.py
│   │   ├── RunExampleApp.py
│   │   ├── ScaleBar.py
│   │   ├── ScatterPlot.py
│   │   ├── ScatterPlotSpeedTest.py
│   │   ├── ScatterPlotWidget.py
│   │   ├── scrollingPlots.py
│   │   ├── SimplePlot.py
│   │   ├── SpinBox.py
│   │   ├── Symbols.py
│   │   ├── syntax.py
│   │   ├── TableWidget.py
│   │   ├── template.py
│   │   ├── test_examples.py
│   │   ├── text.py
│   │   ├── TreeWidget.py
│   │   ├── utils.py
│   │   ├── verlet_chain_demo.py
│   │   ├── VideoSpeedTest.py
│   │   ├── VideoTemplate.py
│   │   ├── VideoTemplate_generic.py
│   │   ├── ViewBox.py
│   │   ├── ViewBoxFeatures.py
│   │   └── ViewLimits.py
│   ├── exporters/
│   │   ├── __init__.py
│   │   ├── CSVExporter.py
│   │   ├── Exporter.py
│   │   ├── HDF5Exporter.py
│   │   ├── ImageExporter.py
│   │   ├── Matplotlib.py
│   │   ├── PrintExporter.py
│   │   └── SVGExporter.py
│   ├── flowchart/
│   │   ├── library/
│   │   │   ├── __init__.py
│   │   │   ├── common.py
│   │   │   ├── Data.py
│   │   │   ├── Display.py
│   │   │   ├── Filters.py
│   │   │   ├── functions.py
│   │   │   └── Operators.py
│   │   ├── __init__.py
│   │   ├── Flowchart.py
│   │   ├── FlowchartCtrlTemplate.py
│   │   ├── FlowchartCtrlTemplate_generic.py
│   │   ├── FlowchartGraphicsView.py
│   │   ├── Node.py
│   │   ├── NodeLibrary.py
│   │   └── Terminal.py
│   ├── graphicsItems/
│   │   ├── PlotItem/
│   │   │   ├── __init__.py
│   │   │   ├── plotConfigTemplate.py
│   │   │   ├── plotConfigTemplate_generic.py
│   │   │   └── PlotItem.py
│   │   ├── ViewBox/
│   │   │   ├── __init__.py
│   │   │   ├── axisCtrlTemplate.py
│   │   │   ├── axisCtrlTemplate_generic.py
│   │   │   ├── ViewBox.py
│   │   │   └── ViewBoxMenu.py
│   │   ├── __init__.py
│   │   ├── ArrowItem.py
│   │   ├── AxisItem.py
│   │   ├── BarGraphItem.py
│   │   ├── BoxplotItem.py
│   │   ├── ButtonItem.py
│   │   ├── ColorBarItem.py
│   │   ├── CurvePoint.py
│   │   ├── DateAxisItem.py
│   │   ├── ErrorBarItem.py
│   │   ├── FillBetweenItem.py
│   │   ├── GradientEditorItem.py
│   │   ├── GradientLegend.py
│   │   ├── GradientPresets.py
│   │   ├── GraphicsItem.py
│   │   ├── GraphicsLayout.py
│   │   ├── GraphicsObject.py
│   │   ├── GraphicsWidget.py
│   │   ├── GraphicsWidgetAnchor.py
│   │   ├── GraphItem.py
│   │   ├── GridItem.py
│   │   ├── HistogramLUTItem.py
│   │   ├── ImageItem.py
│   │   ├── InfiniteLine.py
│   │   ├── IsocurveItem.py
│   │   ├── ItemGroup.py
│   │   ├── LabelItem.py
│   │   ├── LegendItem.py
│   │   ├── LinearRegionItem.py
│   │   ├── NonUniformImage.py
│   │   ├── PColorMeshItem.py
│   │   ├── PlotCurveItem.py
│   │   ├── PlotDataItem.py
│   │   ├── ROI.py
│   │   ├── ScaleBar.py
│   │   ├── ScatterPlotItem.py
│   │   ├── TargetItem.py
│   │   ├── TextItem.py
│   │   ├── UIGraphicsItem.py
│   │   └── VTickGroup.py
│   ├── GraphicsScene/
│   │   ├── __init__.py
│   │   ├── exportDialog.py
│   │   ├── exportDialogTemplate.py
│   │   ├── exportDialogTemplate_generic.py
│   │   ├── GraphicsScene.py
│   │   └── mouseEvents.py
│   ├── icons/
│   │   ├── peegee/
│   │   │   ├── peegee.svg
│   │   │   ├── peegee_128px.png
│   │   │   ├── peegee_128px@2x.png
│   │   │   ├── peegee_192px.png
│   │   │   ├── peegee_256px.png
│   │   │   ├── peegee_256px@2x.png
│   │   │   ├── peegee_512px.png
│   │   │   └── peegee_512px@2x.png
│   │   ├── __init__.py
│   │   ├── auto.png
│   │   ├── ctrl.png
│   │   ├── default.png
│   │   ├── delete.png
│   │   ├── icons.svg
│   │   ├── invisibleEye.svg
│   │   ├── lock.png
│   │   ├── rename.png
│   │   ├── revert_default.png
│   │   ├── set_default.png
│   │   ├── unlock.png
│   │   └── visibleEye.svg
│   ├── imageview/
│   │   ├── __init__.py
│   │   ├── ImageView.py
│   │   ├── ImageViewTemplate.py
│   │   └── ImageViewTemplate_generic.py
│   ├── jupyter/
│   │   ├── __init__.py
│   │   └── GraphicsView.py
│   ├── multiprocess/
│   │   ├── __init__.py
│   │   ├── bootstrap.py
│   │   ├── parallelizer.py
│   │   ├── processes.py
│   │   └── remoteproxy.py
│   ├── opengl/
│   │   ├── items/
│   │   │   ├── __init__.py
│   │   │   ├── GLAxisItem.py
│   │   │   ├── GLBarGraphItem.py
│   │   │   ├── GLBoxItem.py
│   │   │   ├── GLGradientLegendItem.py
│   │   │   ├── GLGraphItem.py
│   │   │   ├── GLGridAxisItem.py
│   │   │   ├── GLGridItem.py
│   │   │   ├── GLImageItem.py
│   │   │   ├── GLLinePlotItem.py
│   │   │   ├── GLMeshItem.py
│   │   │   ├── GLScatterPlotItem.py
│   │   │   ├── GLSurfacePlotItem.py
│   │   │   ├── GLTextItem.py
│   │   │   └── GLVolumeItem.py
│   │   ├── __init__.py
│   │   ├── GLGraphicsItem.py
│   │   ├── GLViewWidget.py
│   │   ├── MeshData.py
│   │   └── shaders.py
│   ├── parametertree/
│   │   ├── parameterTypes/
│   │   │   ├── __init__.py
│   │   │   ├── action.py
│   │   │   ├── actiongroup.py
│   │   │   ├── basetypes.py
│   │   │   ├── bool.py
│   │   │   ├── brush.py
│   │   │   ├── calendar.py
│   │   │   ├── checklist.py
│   │   │   ├── color.py
│   │   │   ├── colormap.py
│   │   │   ├── colormaplut.py
│   │   │   ├── file.py
│   │   │   ├── font.py
│   │   │   ├── list.py
│   │   │   ├── numeric.py
│   │   │   ├── pen.py
│   │   │   ├── progress.py
│   │   │   ├── qtenum.py
│   │   │   ├── slider.py
│   │   │   ├── str.py
│   │   │   └── text.py
│   │   ├── __init__.py
│   │   ├── interactive.py
│   │   ├── iojson.py
│   │   ├── Parameter.py
│   │   ├── ParameterItem.py
│   │   ├── ParameterSystem.py
│   │   ├── ParameterTree.py
│   │   ├── SystemSolver.py
│   │   └── utils.py
│   ├── Qt/
│   │   ├── compat/
│   │   │   └── __init__.py
│   │   ├── QtCore/
│   │   │   ├── __init__.py
│   │   │   └── __init__.pyi
│   │   ├── QtGui/
│   │   │   ├── __init__.py
│   │   │   └── __init__.pyi
│   │   ├── QtOpenGL/
│   │   │   └── __init__.py
│   │   ├── QtWidgets/
│   │   │   ├── __init__.py
│   │   │   └── __init__.pyi
│   │   ├── __init__.py
│   │   ├── __init__.pyi
│   │   ├── internals.py
│   │   ├── OpenGLConstants.py
│   │   ├── OpenGLHelpers.py
│   │   ├── QtSvg.pyi
│   │   └── QtTest.pyi
│   ├── util/
│   │   ├── __init__.py
│   │   ├── cprint.py
│   │   ├── cupy_helper.py
│   │   ├── garbage_collector.py
│   │   ├── get_resolution.py
│   │   ├── glinfo.py
│   │   ├── mutex.py
│   │   └── numba_helper.py
│   ├── widgets/
│   │   ├── __init__.py
│   │   ├── BrushPreviewLabel.py
│   │   ├── BusyCursor.py
│   │   ├── CheckTable.py
│   │   ├── ColorButton.py
│   │   ├── ColorMapButton.py
│   │   ├── ColorMapMenu.py
│   │   ├── ColorMapWidget.py
│   │   ├── ComboBox.py
│   │   ├── DataFilterWidget.py
│   │   ├── DataTreeWidget.py
│   │   ├── DiffTreeWidget.py
│   │   ├── FeedbackButton.py
│   │   ├── FileDialog.py
│   │   ├── GradientWidget.py
│   │   ├── GraphicsLayoutWidget.py
│   │   ├── GraphicsView.py
│   │   ├── GroupBox.py
│   │   ├── HistogramLUTWidget.py
│   │   ├── JoystickButton.py
│   │   ├── LayoutWidget.py
│   │   ├── MatplotlibWidget.py
│   │   ├── PathButton.py
│   │   ├── PenPreviewLabel.py
│   │   ├── PlotWidget.py
│   │   ├── ProgressDialog.py
│   │   ├── RawImageWidget.py
│   │   ├── RemoteGraphicsView.py
│   │   ├── ScatterPlotWidget.py
│   │   ├── SpinBox.py
│   │   ├── TableWidget.py
│   │   ├── TreeWidget.py
│   │   ├── ValueLabel.py
│   │   └── VerticalLabel.py
│   ├── __init__.py
│   ├── algorithms.py
│   ├── arraytoline.py
│   ├── colormap.py
│   ├── configfile.py
│   ├── debug.py
│   ├── exceptionHandling.py
│   ├── frozenSupport.py
│   ├── functions.py
│   ├── functions_numba.py
│   ├── functions_qimage.py
│   ├── PlotData.py
│   ├── Point.py
│   ├── reload.py
│   ├── SignalProxy.py
│   ├── SRTTransform.py
│   ├── SRTTransform3D.py
│   ├── ThreadsafeTimer.py
│   ├── Transform3D.py
│   ├── units.py
│   ├── Vector.py
│   └── WidgetGroup.py
├── tests/
│   ├── dockarea/
│   │   ├── test_dock.py
│   │   └── test_dockarea.py
│   ├── exporters/
│   │   ├── __init__.py
│   │   ├── test_csv.py
│   │   ├── test_exporter_dialog.py
│   │   ├── test_hdf5.py
│   │   ├── test_image.py
│   │   ├── test_matplotlib.py
│   │   └── test_svg.py
│   ├── flowchart/
│   │   └── test_flowchart.py
│   ├── graphicsItems/
│   │   ├── PlotItem/
│   │   │   └── test_PlotItem.py
│   │   ├── ViewBox/
│   │   │   ├── test_ViewBox.py
│   │   │   └── test_ViewBoxZoom.py
│   │   ├── test_ArrowItem.py
│   │   ├── test_AxisItem.py
│   │   ├── test_DateAxisItem.py
│   │   ├── test_ErrorBarItem.py
│   │   ├── test_GraphicsItem.py
│   │   ├── test_ImageItem.py
│   │   ├── test_InfiniteLine.py
│   │   ├── test_LegendItem.py
│   │   ├── test_LinearRegionItem.py
│   │   ├── test_NonUniformImage.py
│   │   ├── test_PlotCurveItem.py
│   │   ├── test_PlotDataItem.py
│   │   ├── test_ROI.py
│   │   ├── test_ScatterPlotItem.py
│   │   └── test_TextItem.py
│   ├── images/
│   │   ├── imageitem/
│   │   │   ├── bool.png
│   │   │   ├── gradient_mono_byte.png
│   │   │   ├── gradient_mono_byte_levels.png
│   │   │   ├── gradient_mono_int.png
│   │   │   ├── gradient_mono_int_levels.png
│   │   │   ├── gradient_rgba_byte.png
│   │   │   ├── gradient_rgba_byte_levels.png
│   │   │   ├── gradient_rgba_float.png
│   │   │   ├── gradient_rgba_float_additive.png
│   │   │   ├── gradient_rgba_float_alpha.png
│   │   │   ├── init.png
│   │   │   ├── levels1.png
│   │   │   ├── lut.png
│   │   │   ├── monochrome.png
│   │   │   ├── resolution_with_downsampling_x.png
│   │   │   ├── resolution_with_downsampling_y.png
│   │   │   └── resolution_without_downsampling.png
│   │   ├── nonuniform_image/
│   │   │   ├── colormap-3x3.png
│   │   │   └── lut-3x3.png
│   │   └── roi/
│   │       ├── baseroi/
│   │       │   ├── roi_getarrayregion.png
│   │       │   ├── roi_getarrayregion_anisotropic.png
│   │       │   ├── roi_getarrayregion_halfpx.png
│   │       │   ├── roi_getarrayregion_img_trans.png
│   │       │   ├── roi_getarrayregion_inverty.png
│   │       │   ├── roi_getarrayregion_resize.png
│   │       │   └── roi_getarrayregion_rotate.png
│   │       ├── ellipseroi/
│   │       │   ├── roi_getarrayregion.png
│   │       │   ├── roi_getarrayregion_anisotropic.png
│   │       │   ├── roi_getarrayregion_halfpx.png
│   │       │   ├── roi_getarrayregion_img_trans.png
│   │       │   ├── roi_getarrayregion_inverty.png
│   │       │   ├── roi_getarrayregion_resize.png
│   │       │   └── roi_getarrayregion_rotate.png
│   │       ├── polylineroi/
│   │       │   ├── closed_clear.png
│   │       │   ├── closed_click_segment.png
│   │       │   ├── closed_drag_handle.png
│   │       │   ├── closed_drag_new_handle.png
│   │       │   ├── closed_drag_roi.png
│   │       │   ├── closed_hover_handle.png
│   │       │   ├── closed_hover_roi.png
│   │       │   ├── closed_hover_segment.png
│   │       │   ├── closed_init.png
│   │       │   ├── closed_setpoints.png
│   │       │   ├── closed_setstate.png
│   │       │   ├── open_clear.png
│   │       │   ├── open_click_segment.png
│   │       │   ├── open_drag_handle.png
│   │       │   ├── open_drag_new_handle.png
│   │       │   ├── open_drag_roi.png
│   │       │   ├── open_hover_handle.png
│   │       │   ├── open_hover_roi.png
│   │       │   ├── open_hover_segment.png
│   │       │   ├── open_init.png
│   │       │   ├── open_setpoints.png
│   │       │   ├── open_setstate.png
│   │       │   ├── roi_getarrayregion.png
│   │       │   ├── roi_getarrayregion_anisotropic.png
│   │       │   ├── roi_getarrayregion_halfpx.png
│   │       │   ├── roi_getarrayregion_img_trans.png
│   │       │   ├── roi_getarrayregion_inverty.png
│   │       │   └── roi_getarrayregion_rotate.png
│   │       └── rectroi/
│   │           ├── roi_getarrayregion.png
│   │           ├── roi_getarrayregion_anisotropic.png
│   │           ├── roi_getarrayregion_halfpx.png
│   │           ├── roi_getarrayregion_img_trans.png
│   │           ├── roi_getarrayregion_inverty.png
│   │           ├── roi_getarrayregion_resize.png
│   │           └── roi_getarrayregion_rotate.png
│   ├── imageview/
│   │   └── test_imageview.py
│   ├── opengl/
│   │   └── items/
│   │       ├── common.py
│   │       ├── test_GLAxisItem.py
│   │       ├── test_GLBarGraphItem.py
│   │       ├── test_GLBoxItem.py
│   │       ├── test_GLGradientLegendItem.py
│   │       ├── test_GLGraphItem.py
│   │       ├── test_GLGridItem.py
│   │       ├── test_GLImageItem.py
│   │       ├── test_GLLinePlotItem.py
│   │       ├── test_GLMeshItem.py
│   │       ├── test_GLScatterPlotItem.py
│   │       ├── test_GLSurfacePlotItem.py
│   │       ├── test_GLTextItem.py
│   │       └── test_GLVolumeItem.py
│   ├── parametertree/
│   │   ├── test_Parameter.py
│   │   ├── test_parametertypes.py
│   │   └── test_utils.py
│   ├── widgets/
│   │   ├── test_busycursor.py
│   │   ├── test_colorbutton.py
│   │   ├── test_combobox.py
│   │   ├── test_graphics_view.py
│   │   ├── test_histogramlutwidget.py
│   │   ├── test_joystickbutton.py
│   │   ├── test_matplotlibwidget.py
│   │   ├── test_progressdialog.py
│   │   ├── test_spinbox.py
│   │   └── test_tablewidget.py
│   ├── __init__.py
│   ├── conftest.py
│   ├── image_testing.py
│   ├── makeARGB_test_data.py
│   ├── test_arraytoline.py
│   ├── test_colormap.py
│   ├── test_configparser.py
│   ├── test_functions.py
│   ├── test_ImageFormat.py
│   ├── test_makeARGB.py
│   ├── test_pickles.py
│   ├── test_Point.py
│   ├── test_qimage_writethru.py
│   ├── test_qmenu_leak_workaround.py
│   ├── test_qpainterpathprivate.py
│   ├── test_qt.py
│   ├── test_ref_cycles.py
│   ├── test_reload.py
│   ├── test_signalproxy.py
│   ├── test_srttransform3d.py
│   ├── test_stability.py
│   ├── test_Vector.py
│   ├── ui_testing.py
│   └── uictest.ui
├── tools/
│   ├── rebuildPtreeRst.py
│   └── rebuildUi.py
├── CHANGELOG
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE.txt
├── pyproject.toml
├── README.md
├── test.py
└── tox.ini
```
