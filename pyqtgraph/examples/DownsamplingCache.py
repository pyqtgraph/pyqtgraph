import argparse
import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QToolButton,
    QWidget,
    QLabel,
    QSpinBox,
)
from PyQt6 import QtCore

import pyqtgraph as pg


class wait_cursor:
    def __enter__(self):
        QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)

    def __exit__(self, exc_type, exc_value, traceback):
        QApplication.restoreOverrideCursor()


class TimeSeriesPlot(QMainWindow):
    def __init__(self, signal_length, n_signals, 
                 use_cache, cache_ds_factor, autostart_zoom_sequence):
        super().__init__()        
        self.setWindowTitle("Downsampling Cache Demo")
        self.setGeometry(100, 100, 800, 500)

        # Main widget and layout
        main_widget = QWidget()
        layout = QVBoxLayout()

        self.items = []        
        self._use_cache = use_cache         
        self._cache_ds_factor = cache_ds_factor


        # PlotWidget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setClipToView(True)
        self.plot_widget.setDownsampling(ds=1, auto=True, mode="peak")
        layout.addWidget(self.plot_widget)

        # Cache controls layout
        cache_layout = QHBoxLayout()
        
        # Use cache tool button
        self.use_cache_button = QToolButton()       
        self.use_cache_button.setText("Use downsampling cache")
        self.use_cache_button.setCheckable(True)
        self.use_cache_button.setChecked(self._use_cache)
        self.use_cache_button.clicked.connect(self.on_use_cache_toggled)
        self.cache_label = QLabel()
        self.on_use_cache_toggled()
        
        # Cache downsampling spinbox
        cache_ds_factor_label = QLabel("Cache downsampling factor:")
        self.cache_ds_factor_spinbox = QSpinBox()
        self.cache_ds_factor_spinbox.setRange(1, 1000000)
        self.cache_ds_factor_spinbox.setValue(self._cache_ds_factor)
        self.cache_ds_factor_spinbox.setSingleStep(1000)
        self.cache_ds_factor_spinbox.setToolTip("Downsampling factor for cache")
        self.cache_ds_factor_spinbox.editingFinished.connect(self.on_cache_ds_changed)        
        
        cache_layout.addWidget(self.use_cache_button)
        cache_layout.addSpacing(10)
        cache_layout.addWidget(cache_ds_factor_label)
        cache_layout.addWidget(self.cache_ds_factor_spinbox)
        cache_layout.addStretch()  # Push everything to the left
        
        cache_widget = QWidget()
        cache_widget.setLayout(cache_layout)
        
        # Recalculate button and spinboxes in horizontal layout
        recalc_layout = QHBoxLayout()
        self.recalc_button = QToolButton()
        self.recalc_button.setText("Recalculate plots")
        self.recalc_button.clicked.connect(self.recalculate_plots)
        
        # Number of signals spinbox
        signals_label = QLabel("Signals:")
        self.num_signals_spinbox = QSpinBox()
        self.num_signals_spinbox.setRange(1, 50)
        self.num_signals_spinbox.setValue(n_signals)
        self.num_signals_spinbox.setToolTip("Number of signals to plot")
        
        # Signal length spinbox
        length_label = QLabel("Length:")
        self.signal_length_spinbox = QSpinBox()
        self.signal_length_spinbox.setRange(1000, 1000000000)
        self.signal_length_spinbox.setValue(signal_length)
        self.signal_length_spinbox.setSingleStep(1000000)
        self.signal_length_spinbox.setToolTip("Length of each signal")
        

        recalc_layout.addWidget(self.recalc_button)
        recalc_layout.addSpacing(20)  # Add some space after button
        recalc_layout.addWidget(signals_label)
        recalc_layout.addWidget(self.num_signals_spinbox)
        recalc_layout.addSpacing(10)  # Add space between spinboxes
        recalc_layout.addWidget(length_label)
        recalc_layout.addWidget(self.signal_length_spinbox)
        recalc_layout.addStretch()  # Push everything to the left
        
        recalc_widget = QWidget()
        recalc_widget.setLayout(recalc_layout)
        
        self.autozoom_button = QToolButton()
        self.autozoom_button.setText("Run zoom sequence")
        self.autozoom_button.clicked.connect(self.autozoom)
        layout.addWidget(self.cache_label)
        layout.addWidget(cache_widget)
        layout.addWidget(recalc_widget)
        layout.addWidget(self.autozoom_button)

        # Apply layout
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        # Plot data
        self.plot_random_time_series(n_signals, signal_length)

        if autostart_zoom_sequence:
            QtCore.QTimer.singleShot(0, self.autozoom)

    def recalculate_plots(self):
        """Recalculate plots using values from spinboxes"""
        num_signals = self.num_signals_spinbox.value()
        signal_length = self.signal_length_spinbox.value()
        self.plot_random_time_series(num_signals, signal_length)

    def plot_random_time_series(self, num_lines, length):
        with wait_cursor():
            self.plot_widget.clear()
            x = np.arange(0, length)
            self.items = []
            for i in range(num_lines):
                y = np.random.default_rng().normal(size=length)
                color = pg.intColor(i, hues=num_lines)
                item = self.plot_widget.plot(x, y, useDownsamplingCache=self._use_cache, minSampPerPxForCache = 1.1, clear=False, pen=pg.mkPen(color))
                self.items.append(item)

    def on_use_cache_toggled(self):
        with wait_cursor():
            for item in self.items:
                item.setDownsamplingCacheMode(useCache=self.use_cache_button.isChecked(), cacheDsFactor=self._cache_ds_factor)
        if self.use_cache_button.isChecked():
            self.cache_label.setText("Using cache")
            self.cache_label.setStyleSheet("color: green")
            self._use_cache = True
        else:
            self.cache_label.setText("Not using cache")
            self.cache_label.setStyleSheet("color: red")
            self._use_cache = False
    
    def on_cache_ds_changed(self):
        """Update cache size when spinbox value changes (if cache is currently enabled)"""
        if self.use_cache_button.isChecked():
            with wait_cursor():
                self._cache_ds_factor = self.cache_ds_factor_spinbox.value()
                for item in self.items:
                    item.setDownsamplingCacheMode(useCache=True, cacheDsFactor=self._cache_ds_factor)
    
    def autozoom(self):
        # Gradually zoom all the way in on the x-axis, then all the way out
        vb = self.plot_widget.getViewBox()
        
        vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
        if self.items:
            xdata = self.items[0].xData
            start = xdata[0]
            end = xdata[-1]
            steps = 100
            # Zoom in
            for i in range(steps):
                frac = i / steps
                rng = end - start
                zoom_width = max(1, rng * (1 - frac)**6)
                center = start + rng / 2
                vb.setXRange(center - zoom_width / 2, center + zoom_width / 2, padding=0)
                QApplication.processEvents()
            # Zoom out
            for i in range(steps):
                frac = i / steps
                rng = end - start
                zoom_width = max(1, rng * frac**6)
                center = start + rng / 2
                vb.setXRange(center - zoom_width / 2, center + zoom_width / 2, padding=0)
                QApplication.processEvents()
            # Ensure fully zoomed out at end
            vb.setXRange(start, end, padding=0)


def parse_args():
    """Parse command line arguments for the TimeSeriesPlot constructor."""
    parser = argparse.ArgumentParser(
        description="Demo PlotDataItem downsampling cache performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--signal-length', '-l',
        type=int,
        default=500_000_000, #Huge signal tha actually shows benefit of cache
        help='Length of each signal in data points'
    )    
    parser.add_argument(
        '--n-signals', '-n',
        type=int,
        default=1,
        help='Number of signals to plot'
    )    
    parser.add_argument(
        '--use-cache',
        default=False,
        action='store_true',
        help='Start with downsampling cache enabled'
    )    
    parser.add_argument(
        '--cache-ds-factor',
        type=int,
        default=20000,
        help='Downsampling factor for cache'
    )    
    parser.add_argument(
        '--autostart-zoom',
        default=False,
        action='store_true',
        help='Automatically start zoom sequence on startup'
    )    
    return parser.parse_args()


if __name__ == "__main__":
    #Use command line arguments if run from command line
    args = parse_args()
    
    pg.mkQApp("Downsampling Cache Demo")
    window = TimeSeriesPlot(
        signal_length=args.signal_length,
        n_signals=args.n_signals,
        use_cache=args.use_cache,
        cache_ds_factor=args.cache_ds_factor,
        autostart_zoom_sequence=args.autostart_zoom
    )
    window.show()
    pg.exec()
else:
    #If run in test script, use parameters that make it run quickly
    pg.mkQApp("Downsampling Cache Demo")
    window = TimeSeriesPlot(signal_length=100_000, n_signals=2, use_cache=True, cache_ds_factor=200, autostart_zoom_sequence=True)
    window.show()
