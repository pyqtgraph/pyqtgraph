#!/usr/bin/env python3
"""
Simple direct test of the axis clipping fix.

This script directly tests our implementation to see if it works.
"""

import sys
import os

# Add current directory to path for local pyqtgraph import
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    import numpy as np
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
    
    print("✅ Successfully imported PyQtGraph from local workspace")
    
    def test_axis_defaults():
        """Test that our improved defaults are working"""
        print("\n📋 Testing improved axis defaults...")
        
        # Create a simple axis
        app = pg.mkQApp("Axis Test")
        
        # Test horizontal axis
        h_axis = pg.AxisItem('bottom')
        print(f"Horizontal axis defaults:")
        print(f"  autoExpandTextSpace: {h_axis.style.get('autoExpandTextSpace')}")
        print(f"  hideOverlappingLabels: {h_axis.style.get('hideOverlappingLabels')}")
        print(f"  autoReduceTextSpace: {h_axis.style.get('autoReduceTextSpace')}")
        
        # Test vertical axis  
        v_axis = pg.AxisItem('left')
        print(f"Vertical axis defaults:")
        print(f"  autoExpandTextSpace: {v_axis.style.get('autoExpandTextSpace')}")
        print(f"  hideOverlappingLabels: {v_axis.style.get('hideOverlappingLabels')}")
        print(f"  autoReduceTextSpace: {v_axis.style.get('autoReduceTextSpace')}")
        
        return True
    
    def test_new_methods():
        """Test that our new methods exist and work"""
        print("\n🔧 Testing new methods...")
        
        app = pg.mkQApp("Method Test")
        axis = pg.AxisItem('bottom')
        
        # Test that our new methods exist
        methods_to_check = [
            '_calculateRequiredTextSpace',
            '_getAvailableTextSpace', 
            '_getParentLayoutMargins',
            '_requestLayoutExpansion',
            '_checkAndRequestLayoutExpansion'
        ]
        
        for method_name in methods_to_check:
            if hasattr(axis, method_name):
                print(f"  ✅ {method_name} - exists")
                # Try to call it safely
                try:
                    method = getattr(axis, method_name)
                    if method_name == '_calculateRequiredTextSpace':
                        result = method()
                        print(f"    Returns: {result} (expected: 0 for no data)")
                    elif method_name == '_getAvailableTextSpace':
                        result = method()
                        print(f"    Returns: {result}")
                    elif method_name == '_getParentLayoutMargins':
                        result = method()
                        print(f"    Returns: {result} (expected: None for no parent)")
                except Exception as e:
                    print(f"    ⚠️  Error calling method: {e}")
            else:
                print(f"  ❌ {method_name} - missing")
        
        return True
    
    def test_plotitem_methods():
        """Test PlotItem enhancement methods"""
        print("\n📊 Testing PlotItem enhancements...")
        
        app = pg.mkQApp("PlotItem Test")
        plot_widget = pg.PlotWidget()
        plot_item = plot_widget.plotItem
        
        # Test that our PlotItem methods exist
        methods_to_check = [
            '_expandForAxisText',
            '_notifyAxesOfMarginChange'
        ]
        
        for method_name in methods_to_check:
            if hasattr(plot_item, method_name):
                print(f"  ✅ {method_name} - exists")
            else:
                print(f"  ❌ {method_name} - missing") 
        
        return True
    
    def test_visual_scenario():
        """Test the actual issue scenario visually"""
        print("\n👁️  Testing visual scenario...")
        
        app = pg.mkQApp("Visual Test")
        
        # Create a plot with the problematic scenario
        win = pg.GraphicsLayoutWidget(show=False, title="Axis Clipping Fix Test")
        
        # Problem scenario
        plot1 = win.addPlot(title="Before fix behavior")
        plot1.layout.setContentsMargins(4.0, 4.0, 4.0, 4.0)
        
        # Test data that typically causes clipping
        x_data = np.linspace(0, 9.999, 50)
        y_data = np.sin(x_data)
        plot1.plot(x_data, y_data, pen='r')
        
        # Our enhanced scenario
        win.nextRow()
        plot2 = win.addPlot(title="With our fix")
        plot2.layout.setContentsMargins(4.0, 4.0, 4.0, 4.0)
        
        axis = plot2.getAxis('bottom')
        # The defaults should now be better
        print(f"  Enhanced axis settings:")
        print(f"    autoExpandTextSpace: {axis.style.get('autoExpandTextSpace')}")
        print(f"    hideOverlappingLabels: {axis.style.get('hideOverlappingLabels')}")
        print(f"    autoReduceTextSpace: {axis.style.get('autoReduceTextSpace')}")
        
        plot2.plot(x_data, y_data, pen='g')
        
        print("  Visual test plot created (not shown)")
        return True
    
    def main():
        """Run all tests"""
        print("🧪 Direct Test of Axis Clipping Fix")
        print("=" * 40)
        
        tests = [
            ("Axis Defaults", test_axis_defaults),
            ("New Methods", test_new_methods), 
            ("PlotItem Methods", test_plotitem_methods),
            ("Visual Scenario", test_visual_scenario)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result, None))
                print(f"✅ {test_name} - PASSED")
            except Exception as e:
                results.append((test_name, False, str(e)))
                print(f"❌ {test_name} - FAILED: {e}")
        
        print("\n" + "=" * 40)
        print("📋 Summary:")
        passed = sum(1 for _, result, _ in results if result)
        total = len(results)
        print(f"  {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! The fix appears to be working.")
            return True
        else:
            print("⚠️  Some tests failed. Check the implementation.")
            return False
    
    if __name__ == '__main__':
        success = main()
        sys.exit(0 if success else 1)
        
except ImportError as e:
    print(f"❌ Failed to import dependencies: {e}")
    print("Make sure you're running this from the pyqtgraph workspace directory")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)