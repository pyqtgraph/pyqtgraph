#!/usr/bin/env python3
"""
Test script to reproduce the axis text clipping issue from:
https://github.com/pyqtgraph/pyqtgraph/issues/3375

This script demonstrates the problem where rightmost tick labels 
are clipped when using content margins in PlotItem.
"""

import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

def test_original_issue():
    """Reproduce the original issue exactly as reported"""
    print("Testing original issue: rightmost tick labels clipped with content margins")
    
    app = pg.mkQApp("Axis Clipping Test")
    
    # Create the exact scenario from the issue
    win = pg.GraphicsLayoutWidget(show=True, title="Axis Clipping Issue Test")
    win.resize(800, 600)
    
    # Create plot with content margins - this causes the issue
    plot = win.addPlot(title="Problem: Rightmost label clipped")
    plot.layout.setContentsMargins(4.0, 4.0, 4.0, 4.0)
    
    # Add data that creates tick values likely to cause clipping
    x_data = np.linspace(0, 9.999, 100)  # Ending at 9.999 often causes clipping
    y_data = np.sin(x_data)
    plot.plot(x_data, y_data, pen='r', name='Original Issue')
    
    # Print current axis settings
    bottom_axis = plot.getAxis('bottom')
    print(f"Bottom axis settings:")
    print(f"  autoExpandTextSpace: {bottom_axis.style.get('autoExpandTextSpace', 'Not set')}")
    print(f"  hideOverlappingLabels: {bottom_axis.style.get('hideOverlappingLabels', 'Not set')}")
    print(f"  autoReduceTextSpace: {bottom_axis.style.get('autoReduceTextSpace', 'Not set')}")
    print(f"  textWidth: {bottom_axis.textWidth}")
    print(f"  Layout margins: {plot.layout.contentsMargins()}")
    
    # Add another plot with the known workaround for comparison
    win.nextRow()
    plot2 = win.addPlot(title="Workaround: Should work correctly")
    
    # Apply the workaround settings from the issue comment
    bottom_axis2 = plot2.getAxis('bottom')
    bottom_axis2.setStyle(
        hideOverlappingLabels=False,
        autoExpandTextSpace=True,
        autoReduceTextSpace=False
    )
    
    plot2.plot(x_data, y_data, pen='g', name='With Workaround')
    
    print(f"\nWorkaround axis settings:")
    print(f"  autoExpandTextSpace: {bottom_axis2.style.get('autoExpandTextSpace', 'Not set')}")
    print(f"  hideOverlappingLabels: {bottom_axis2.style.get('hideOverlappingLabels', 'Not set')}")
    print(f"  autoReduceTextSpace: {bottom_axis2.style.get('autoReduceTextSpace', 'Not set')}")
    
    return app, win

def test_various_scenarios():
    """Test multiple scenarios that might cause text clipping"""
    print("\nTesting various clipping scenarios...")
    
    app = pg.mkQApp("Extended Clipping Tests")
    
    win = pg.GraphicsLayoutWidget(show=True, title="Extended Clipping Tests")
    win.resize(1200, 800)
    
    scenarios = [
        ("Large numbers", np.linspace(999.99, 1999.99, 50)),
        ("Decimal precision", np.linspace(0, 10.12345, 50)), 
        ("Scientific notation", np.logspace(1, 6, 50)),
        ("Small window", np.linspace(0, 5, 20))
    ]
    
    for i, (title, x_data) in enumerate(scenarios):
        if i % 2 == 0 and i > 0:
            win.nextRow()
            
        y_data = np.random.random(len(x_data))
        
        plot = win.addPlot(title=f"{title} - With margins")
        plot.layout.setContentsMargins(4.0, 4.0, 4.0, 4.0)
        plot.plot(x_data, y_data, pen=pg.mkPen(color=(255, 100, 100)))
        
        # Check if rightmost label would be clipped
        axis = plot.getAxis('bottom')
        axis_rect = axis.boundingRect()
        print(f"{title}: Axis width = {axis_rect.width():.1f}, textWidth = {axis.textWidth}")
    
    return app, win

def test_font_sizes():
    """Test with different font sizes that might cause more clipping"""
    print("\nTesting different font sizes...")
    
    app = pg.mkQApp("Font Size Clipping Tests")
    win = pg.GraphicsLayoutWidget(show=True, title="Font Size Tests")
    win.resize(800, 600)
    
    x_data = np.linspace(0, 12.345, 50)
    y_data = np.sin(x_data)
    
    font_sizes = [8, 10, 12, 14, 16]
    
    for i, size in enumerate(font_sizes):
        if i % 2 == 0 and i > 0:
            win.nextRow()
            
        plot = win.addPlot(title=f"Font size {size}pt")
        plot.layout.setContentsMargins(4.0, 4.0, 4.0, 4.0)
        plot.plot(x_data, y_data, pen=pg.mkPen(color=(100, 255, 100)))
        
        # Set font size for axis
        axis = plot.getAxis('bottom')
        font = QtGui.QFont()
        font.setPointSize(size)
        axis.setStyle(tickFont=font)
        
        print(f"Font {size}pt: textWidth = {axis.textWidth}")
    
    return app, win

def main():
    """Run all test scenarios"""
    print("PyQtGraph Axis Clipping Issue Reproduction Tests")
    print("=" * 50)
    
    # Test the original issue
    app1, win1 = test_original_issue()
    
    # Test various scenarios
    app2, win2 = test_various_scenarios()
    
    # Test font sizes
    app3, win3 = test_font_sizes()
    
    print("\n" + "=" * 50)
    print("All test windows created. Check for rightmost label clipping.")
    print("The first window should show the problem and the workaround.")
    print("\nPress Ctrl+C to exit all windows.")
    
    # Show all windows
    try:
        if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
            app1.exec()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)

if __name__ == '__main__':
    main()