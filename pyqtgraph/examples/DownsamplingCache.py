import argparse
import numpy as np
from pyqtgraph.Qt.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QToolButton,
    QWidget,
    QLabel,
    QSpinBox,
)
from pyqtgraph.Qt import QtCore

import pyqtgraph as pg



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
        self.cache_ds_factor_spinbox.valueChanged.connect(self.on_cache_ds_changed)    
        
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
        
        # Memory usage label
        self.memory_label = QLabel()
        self.update_memory_label()
        self.num_signals_spinbox.valueChanged.connect(self.update_memory_label)
        self.signal_length_spinbox.valueChanged.connect(self.update_memory_label)
        

        recalc_layout.addWidget(self.recalc_button)
        recalc_layout.addSpacing(20)  # Add some space after button
        recalc_layout.addWidget(signals_label)
        recalc_layout.addWidget(self.num_signals_spinbox)
        recalc_layout.addSpacing(10)  # Add space between spinboxes
        recalc_layout.addWidget(length_label)
        recalc_layout.addWidget(self.signal_length_spinbox)
        recalc_layout.addSpacing(10)
        recalc_layout.addWidget(self.memory_label)
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
        with pg.BusyCursor():
            self.plot_widget.clear()
            x = np.arange(0, length)
            self.items = []
            for i in range(num_lines):
                y = np.random.default_rng().normal(size=length)
                color = pg.intColor(i, hues=num_lines)
                item = self.plot_widget.plot(x, y, useDownsamplingCache=self._use_cache, clear=False, pen=pg.mkPen(color))
                self.items.append(item)

    def on_use_cache_toggled(self):
        with pg.BusyCursor():
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
            with pg.BusyCursor():
                self._cache_ds_factor = self.cache_ds_factor_spinbox.value()
                self.plot_widget.setDownsamplingCacheMode(useCache=True, cacheDsFactor=self._cache_ds_factor)

    def update_memory_label(self):
        """Update memory usage label based on current spinbox values"""
        num_signals = self.num_signals_spinbox.value()
        signal_length = self.signal_length_spinbox.value()
        bytes_per_point = 16  # Assuming float64 for x and y
        total_bytes = num_signals * signal_length * bytes_per_point
        if total_bytes < 1e6:
            mem_str = f"{total_bytes / 1e3:.2f} KB"
        elif total_bytes < 1e9:
            mem_str = f"{total_bytes / 1e6:.2f} MB"
        else:
            mem_str = f"{total_bytes / 1e9:.2f} GB"
        self.memory_label.setText(f"Estimated memory usage: {mem_str}")
    
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



if __name__ == "__main__":
        
    pg.mkQApp("Downsampling Cache Demo")
    window = TimeSeriesPlot(
        signal_length=100_000_000,#Use a large signal length to demonstrate cache benefit
        n_signals=1,
        use_cache=False,
        cache_ds_factor=20000,
        autostart_zoom_sequence=False
    )
    window.show()
    pg.exec()
else:
    #If run in test script, use parameters that make it run quickly
    pg.mkQApp("Downsampling Cache Demo")
    window = TimeSeriesPlot(signal_length=100_000, n_signals=2, use_cache=True, cache_ds_factor=200, autostart_zoom_sequence=True)
    window.show()
