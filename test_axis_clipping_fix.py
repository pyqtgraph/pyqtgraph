#!/usr/bin/env python3
"""
Comprehensive test suite for axis text clipping fix.

Tests the implementation that addresses GitHub issue #3375:
https://github.com/pyqtgraph/pyqtgraph/issues/3375

This test suite validates:
1. Backward compatibility
2. Default settings improvements  
3. Text space calculation methods
4. PlotItem layout expansion support
5. Integration between AxisItem and PlotItem
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch
import numpy as np

# Add the current directory to Python path to import local pyqtgraph
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
    from pyqtgraph.graphicsItems.AxisItem import AxisItem
    from pyqtgraph.graphicsItems.PlotItem.PlotItem import PlotItem
    DEPENDENCIES_AVAILABLE = True
    print("✅ PyQtGraph imported successfully from local workspace")
except ImportError as e:
    print(f"Warning: Missing dependencies - {e}")
    DEPENDENCIES_AVAILABLE = False

@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "PyQtGraph dependencies not available")
class TestAxisClippingFix(unittest.TestCase):
    """Test suite for axis text clipping fix"""
    
    def setUp(self):
        """Set up test environment"""
        self.app = pg.mkQApp()
        
    def tearDown(self):
        """Clean up after tests"""
        # Clean up any windows
        try:
            self.app.processEvents()
        except:
            pass

    def test_improved_default_settings(self):
        """Test that default axis settings are improved"""
        
        # Test horizontal axis defaults
        h_axis = AxisItem('bottom')
        self.assertTrue(h_axis.style['autoExpandTextSpace'], 
                       "Horizontal axis should have autoExpandTextSpace=True by default")
        self.assertFalse(h_axis.style['hideOverlappingLabels'], 
                        "Horizontal axis should have hideOverlappingLabels=False to prevent clipping")
        self.assertFalse(h_axis.style['autoReduceTextSpace'],
                        "Horizontal axis should have autoReduceTextSpace=False for better stability")
        
        # Test vertical axis defaults  
        v_axis = AxisItem('left')
        self.assertTrue(v_axis.style['autoExpandTextSpace'],
                       "Vertical axis should have autoExpandTextSpace=True by default")
        self.assertFalse(v_axis.style['hideOverlappingLabels'],
                        "Vertical axis should have hideOverlappingLabels=False")
        self.assertFalse(v_axis.style['autoReduceTextSpace'],
                        "Vertical axis should have autoReduceTextSpace=False for better stability")

    def test_axis_text_space_calculation_methods(self):
        """Test new text space calculation methods in AxisItem"""
        
        axis = AxisItem('bottom')
        
        # Test method existence
        self.assertTrue(hasattr(axis, '_calculateRequiredTextSpace'),
                       "AxisItem should have _calculateRequiredTextSpace method")
        self.assertTrue(hasattr(axis, '_getAvailableTextSpace'),
                       "AxisItem should have _getAvailableTextSpace method")
        self.assertTrue(hasattr(axis, '_getParentLayoutMargins'),
                       "AxisItem should have _getParentLayoutMargins method")
        self.assertTrue(hasattr(axis, '_requestLayoutExpansion'),
                       "AxisItem should have _requestLayoutExpansion method")
        
        # Test method calls don't crash with no data
        try:
            result = axis._calculateRequiredTextSpace()
            self.assertIsInstance(result, (int, float), 
                                "Should return numeric value")
            self.assertGreaterEqual(result, 0, 
                                   "Should return non-negative value")
        except Exception as e:
            self.fail(f"_calculateRequiredTextSpace should not crash: {e}")
            
        try:
            result = axis._getAvailableTextSpace()
            self.assertIsInstance(result, (int, float),
                                "Should return numeric value")
            self.assertGreaterEqual(result, 0,
                                   "Should return non-negative value")
        except Exception as e:
            self.fail(f"_getAvailableTextSpace should not crash: {e}")

    def test_plotitem_expansion_methods(self):
        """Test new layout expansion methods in PlotItem"""
        
        plot_item = PlotItem()
        
        # Test method existence
        self.assertTrue(hasattr(plot_item, '_expandForAxisText'),
                       "PlotItem should have _expandForAxisText method")
        self.assertTrue(hasattr(plot_item, '_notifyAxesOfMarginChange'),
                       "PlotItem should have _notifyAxesOfMarginChange method")
        self.assertTrue(hasattr(plot_item, 'setContentsMargins'),
                       "PlotItem should have setContentsMargins method")
        
        # Test methods don't crash
        try:
            plot_item._expandForAxisText('bottom', 100)
            plot_item._notifyAxesOfMarginChange()
            plot_item.setContentsMargins(5, 5, 5, 5)
        except Exception as e:
            self.fail(f"PlotItem expansion methods should not crash: {e}")

    def test_backward_compatibility(self):
        """Test that existing code still works as before"""
        
        # Test creating plot with old-style usage
        try:
            plot_widget = pg.PlotWidget()
            plot_widget.plot([1, 2, 3], [1, 4, 2])
            
            # Should be able to get axes
            bottom_axis = plot_widget.plotItem.getAxis('bottom')
            left_axis = plot_widget.plotItem.getAxis('left')
            
            self.assertIsInstance(bottom_axis, AxisItem)
            self.assertIsInstance(left_axis, AxisItem)
            
        except Exception as e:
            self.fail(f"Basic plotting should still work: {e}")

    def test_manual_settings_override(self):
        """Test that manual axis settings still override defaults"""
        
        axis = AxisItem('bottom')
        
        # Change settings manually
        axis.setStyle(
            autoExpandTextSpace=False,
            hideOverlappingLabels=True,
            autoReduceTextSpace=True
        )
        
        # Verify settings took effect
        self.assertFalse(axis.style['autoExpandTextSpace'])
        self.assertTrue(axis.style['hideOverlappingLabels'])
        self.assertTrue(axis.style['autoReduceTextSpace'])

    def test_margins_integration(self):
        """Test integration between PlotItem margins and AxisItem"""
        
        plot_item = PlotItem()
        
        # Set content margins
        plot_item.setContentsMargins(10, 10, 10, 10)
        
        # Get an axis and test margin retrieval
        bottom_axis = plot_item.getAxis('bottom')
        margins = bottom_axis._getParentLayoutMargins()
        
        if margins is not None:
            # If we got margins, they should be reasonable
            self.assertGreaterEqual(margins.right(), 0)
            self.assertGreaterEqual(margins.left(), 0)
            self.assertGreaterEqual(margins.top(), 0) 
            self.assertGreaterEqual(margins.bottom(), 0)

    def test_text_space_calculation_with_data(self):
        """Test text space calculation with actual tick data"""
        
        plot_widget = pg.PlotWidget()
        
        # Add data that typically causes clipping issues
        x_data = np.linspace(0, 9.999, 50)
        y_data = np.sin(x_data)
        plot_widget.plot(x_data, y_data)
        
        # Force a layout update
        plot_widget.show()
        plot_widget.resize(400, 300)
        self.app.processEvents()
        
        # Get axis and test calculation
        bottom_axis = plot_widget.plotItem.getAxis('bottom')
        
        # The axis should have reasonable dimensions
        axis_rect = bottom_axis.boundingRect()
        self.assertGreater(axis_rect.width(), 0, "Axis should have positive width")
        self.assertGreater(axis_rect.height(), 0, "Axis should have positive height")
        
        # Test space calculations
        try:
            required_space = bottom_axis._calculateRequiredTextSpace()
            available_space = bottom_axis._getAvailableTextSpace()
            
            self.assertIsInstance(required_space, (int, float))
            self.assertIsInstance(available_space, (int, float))
            self.assertGreaterEqual(required_space, 0)
            self.assertGreaterEqual(available_space, 0)
            
        except Exception as e:
            self.fail(f"Text space calculation should work with real data: {e}")
        
        plot_widget.close()

    def test_font_metrics_compatibility(self):
        """Test compatibility with different Qt versions for font metrics"""
        
        axis = AxisItem('bottom')
        
        # Test with different font sizes
        for size in [8, 10, 12, 16]:
            font = QtGui.QFont()
            font.setPointSize(size)
            axis.setStyle(tickFont=font)
            
            # The axis should handle different font sizes without crashing
            try:
                # This would internally use font metrics
                axis._calculateRequiredTextSpace()
            except Exception as e:
                self.fail(f"Should handle font size {size}: {e}")

    def test_orientation_specific_behavior(self):
        """Test that different axis orientations behave appropriately"""
        
        orientations = ['left', 'right', 'top', 'bottom']
        
        for orientation in orientations:
            with self.subTest(orientation=orientation):
                axis = AxisItem(orientation)
                
                # Each orientation should have appropriate defaults
                self.assertTrue(axis.style['autoExpandTextSpace'])
                self.assertFalse(axis.style['hideOverlappingLabels'])
                self.assertFalse(axis.style['autoReduceTextSpace'])
                
                # Methods should work for all orientations
                try:
                    axis._calculateRequiredTextSpace()
                    axis._getAvailableTextSpace()
                except Exception as e:
                    self.fail(f"Methods should work for {orientation}: {e}")

    def test_error_handling(self):
        """Test that error conditions are handled gracefully"""
        
        axis = AxisItem('bottom')
        plot_item = PlotItem()
        
        # Test with invalid/extreme values
        try:
            plot_item._expandForAxisText('invalid_orientation', -100)
            plot_item._expandForAxisText('bottom', float('inf'))
            plot_item._expandForAxisText('bottom', float('nan'))
        except Exception as e:
            self.fail(f"Should handle invalid values gracefully: {e}")
        
        # Test with disconnected axis
        try:
            disconnected_axis = AxisItem('bottom')
            disconnected_axis._calculateRequiredTextSpace()
            disconnected_axis._requestLayoutExpansion(100)
        except Exception as e:
            self.fail(f"Should handle disconnected axis gracefully: {e}")


class TestAxisClippingIntegration(unittest.TestCase):
    """Integration tests for the complete axis clipping fix"""
    
    def setUp(self):
        """Set up test environment"""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest("PyQtGraph dependencies not available")
        self.app = pg.mkQApp()

    def test_original_issue_scenario(self):
        """Test the exact scenario from GitHub issue #3375"""
        
        try:
            # Create the exact setup from the issue
            layout = pg.GraphicsLayoutWidget()
            plot = layout.addPlot()
            plot.layout.setContentsMargins(4.0, 4.0, 4.0, 4.0)
            
            # Add data that typically causes the issue
            x_data = np.linspace(0, 9.999, 100)
            y_data = np.sin(x_data)
            plot.plot(x_data, y_data)
            
            # Show and process
            layout.show()
            layout.resize(400, 300)
            self.app.processEvents()
            
            # The axis should now handle text space better
            bottom_axis = plot.getAxis('bottom')
            
            # With our fix, these should be the improved defaults
            self.assertTrue(bottom_axis.style.get('autoExpandTextSpace', False))
            self.assertFalse(bottom_axis.style.get('hideOverlappingLabels', True))
            
            # And the axis should have reasonable bounds
            axis_rect = bottom_axis.boundingRect()
            self.assertGreater(axis_rect.width(), 0)
            
            layout.close()
            
        except Exception as e:
            self.fail(f"Original issue scenario should work: {e}")

    def test_workaround_compatibility(self):
        """Test that the known workaround still works"""
        
        try:
            # Test the workaround from the issue comments
            layout = pg.GraphicsLayoutWidget()
            plot = layout.addPlot()
            
            # Apply the workaround settings
            bottom_axis = plot.getAxis('bottom')
            bottom_axis.setStyle(
                hideOverlappingLabels=False,
                autoExpandTextSpace=True,
                autoReduceTextSpace=False
            )
            
            # Add challenging data
            x_data = np.linspace(0, 12.345678, 50)
            y_data = np.random.random(50)
            plot.plot(x_data, y_data)
            
            layout.show()
            layout.resize(600, 400)
            self.app.processEvents()
            
            # Should work without issues
            self.assertTrue(bottom_axis.style['autoExpandTextSpace'])
            self.assertFalse(bottom_axis.style['hideOverlappingLabels'])
            
            layout.close()
            
        except Exception as e:
            self.fail(f"Workaround should still be compatible: {e}")


def run_visual_tests():
    """Run visual tests that require human verification"""
    if not DEPENDENCIES_AVAILABLE:
        print("Cannot run visual tests - dependencies not available")
        return
        
    print("Running visual tests...")
    print("These tests will show windows - check that rightmost labels are not clipped")
    
    app = pg.mkQApp()
    
    # Test 1: Original issue recreation
    win1 = pg.GraphicsLayoutWidget(show=True, title="Test 1: Should NOT clip rightmost label")
    plot1 = win1.addPlot(title="With margins (fixed)")
    plot1.layout.setContentsMargins(4.0, 4.0, 4.0, 4.0)
    x_data = np.linspace(0, 9.999, 100)
    y_data = np.sin(x_data)
    plot1.plot(x_data, y_data, pen='r')
    
    # Test 2: Challenging values
    win2 = pg.GraphicsLayoutWidget(show=True, title="Test 2: Challenging values")
    plot2 = win2.addPlot(title="Large numbers with margins")
    plot2.layout.setContentsMargins(5.0, 5.0, 5.0, 5.0)
    x_data2 = np.linspace(999.99, 1999.99, 50)
    y_data2 = np.random.random(50)
    plot2.plot(x_data2, y_data2, pen='g')
    
    print("Visual tests launched. Close windows when done checking.")
    print("Expected: Rightmost tick labels should be fully visible in both windows.")
    
    return app, [win1, win2]


if __name__ == '__main__':
    # Run unit tests
    print("Running axis clipping fix test suite...")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTest(unittest.makeSuite(TestAxisClippingFix))
    test_suite.addTest(unittest.makeSuite(TestAxisClippingIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
        
        # Ask if user wants to run visual tests
        try:
            response = input("\nRun visual tests? (y/n): ").lower().strip()
            if response.startswith('y'):
                app, windows = run_visual_tests()
                if app:
                    app.exec()
        except (KeyboardInterrupt, EOFError):
            print("\nSkipping visual tests.")
            
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        for test, traceback in result.failures + result.errors:
            print(f"\nFAILED: {test}")
            print(traceback)
        
        sys.exit(1)
    
    print("Test suite completed.")