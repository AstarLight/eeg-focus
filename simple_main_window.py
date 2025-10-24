"""
Simplified Main Window Module
Uses matplotlib instead of pyqtgraph to reduce dependencies
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox, 
                             QFileDialog, QGroupBox, QGridLayout, QSlider,
                             QTextEdit, QSplitter, QCheckBox, QScrollArea, QSizePolicy)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread, QSize
from PyQt5.QtGui import QFont, QPalette, QColor
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5 backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from eeg_processor import EEGProcessor

# Set font (removed Chinese font settings)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


class MatplotlibPlotWidget(QWidget):
    """Matplotlib-based plotting widget"""
    
    def __init__(self, title="", ylabel="", xlabel=""):
        super().__init__()
        self.figure = Figure(figsize=(10, 3))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        self.ax.set_title(title)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel)
        self.ax.grid(True, alpha=0.3)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Initialize data
        self.time_data = np.linspace(0, 2, 500)
        self.signal_data = np.zeros(500)
        self.line, = self.ax.plot(self.time_data, self.signal_data, 'b-', linewidth=2)
        
        self.ax.set_xlim(0, 2)
        self.ax.set_ylim(-200, 200)  # Adjusted to microvolt level range
    
    def update_signal(self, data):
        """Update signal display"""
        if len(data) == 0:
            return
        
        # Adjust data length
        if len(data) > len(self.time_data):
            data = data[:len(self.time_data)]
        elif len(data) < len(self.time_data):
            padded_data = np.zeros(len(self.time_data))
            padded_data[:len(data)] = data
            data = padded_data
        
        self.signal_data = data
        self.line.set_ydata(self.signal_data)
        
        # Dynamically adjust Y-axis range
        if len(data) > 0:
            y_min, y_max = np.min(data), np.max(data)
            margin = (y_max - y_min) * 0.1 + 0.1
            self.ax.set_ylim(y_min - margin, y_max + margin)
        
        self.canvas.draw()
    
    def clear_signal(self):
        """Clear signal display"""
        self.signal_data = np.zeros(len(self.time_data))
        self.line.set_ydata(self.signal_data)
        self.ax.set_ylim(-200, 200)  # Adjusted to microvolt level range
        self.canvas.draw()


class CombinedSignalWidget(QWidget):
    """Combined signal display widget - Original signal and frequency band signals on the same plot"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        
        # Initialize data and color definitions
        self.time_data = np.linspace(0, 5, 1250)  # 5 seconds window with 1250 points
        self.lines = {}
        self.visible_signals = {}
        
        # Define signal colors
        self.colors = {
            'Original Signal': 'black',
            'Delta': 'purple',
            'Theta': 'blue', 
            'Alpha': 'green',
            'Beta': 'orange',
            'Gamma': 'red'
        }
        
        # Default display original signal and Beta wave
        self.visible_signals = {
            'Original Signal': True,
            'Delta': False,
            'Theta': False,
            'Alpha': False,
            'Beta': True,
            'Gamma': False
        }
        
        # Create control panel
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # Create plotting widget
        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        self.ax.set_title("EEG Signal Waveform Analysis")
        self.ax.set_ylabel("Amplitude (μV)")
        self.ax.set_xlabel("Time (s)")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_facecolor('white')
        self.figure.patch.set_facecolor('white')
        
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Initialize all signal lines
        self._init_lines()
        
        self._update_line_visibility()
        self.ax.set_xlim(0, 5)  # 5 seconds window
        self.ax.set_ylim(-200, 200)  # Adjusted to microvolt level range
        
    def create_control_panel(self):
        """Create control panel"""
        panel = QWidget()
        layout = QHBoxLayout()
        
        # Add checkboxes
        self.checkboxes = {}
        signal_names = ['Original Signal', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        
        for name in signal_names:
            checkbox = QCheckBox(name)
            checkbox.setChecked(self.visible_signals.get(name, False))
            checkbox.stateChanged.connect(lambda state, n=name: self.on_checkbox_changed(n, state))
            
            # Set checkbox color
            if name in self.colors:
                checkbox.setStyleSheet(f"QCheckBox {{ color: {self.colors[name]}; font-weight: bold; }}")
            
            self.checkboxes[name] = checkbox
            layout.addWidget(checkbox)
        
        # Add stretch space
        layout.addStretch()
        
        # Add select all/deselect all buttons
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all)
        self.select_all_btn.setMaximumWidth(240)
        layout.addWidget(self.select_all_btn)
        
        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.clicked.connect(self.deselect_all)
        self.deselect_all_btn.setMaximumWidth(240)
        layout.addWidget(self.deselect_all_btn)
        
        panel.setLayout(layout)
        return panel
    
    def _init_lines(self):
        """Initialize all signal lines"""
        for name, color in self.colors.items():
            line, = self.ax.plot([], [], label=name, color=color, linewidth=1.5, alpha=0.8)
            self.lines[name] = line
    
    def on_checkbox_changed(self, signal_name, state):
        """Checkbox state changed"""
        self.visible_signals[signal_name] = (state == Qt.Checked)
        self._update_line_visibility()
        self.canvas.draw()
    
    def select_all(self):
        """Select all signals"""
        for name in self.checkboxes:
            self.checkboxes[name].setChecked(True)
            self.visible_signals[name] = True
        self._update_line_visibility()
        self.canvas.draw()
    
    def deselect_all(self):
        """Deselect all signals"""
        for name in self.checkboxes:
            self.checkboxes[name].setChecked(False)
            self.visible_signals[name] = False
        self._update_line_visibility()
        self.canvas.draw()
    
    def _update_line_visibility(self):
        """Update line visibility"""
        for name, line in self.lines.items():
            visible = self.visible_signals.get(name, False)
            line.set_visible(visible)
        
        # Update legend to show only visible lines
        self._update_legend()
    
    def _update_legend(self):
        """Update legend to show only visible lines with proper line samples"""
        # Get visible lines and their labels
        visible_lines = []
        visible_labels = []
        
        for name, line in self.lines.items():
            if line.get_visible():
                visible_lines.append(line)
                visible_labels.append(name)
        
        # Remove old legend if exists
        if self.ax.get_legend():
            self.ax.get_legend().remove()
        
        # Create new legend with visible lines only
        if visible_lines:
            self.ax.legend(visible_lines, visible_labels, loc='upper right')
    
    def update_signals(self, original_data=None, band_data=None):
        """Update signal display"""
        # Update original signal
        if original_data is not None and len(original_data) > 0:
            # Adjust data length
            if len(original_data) > len(self.time_data):
                original_data = original_data[:len(self.time_data)]
            elif len(original_data) < len(self.time_data):
                padded_data = np.zeros(len(self.time_data))
                padded_data[:len(original_data)] = original_data
                original_data = padded_data
            
            if 'Original Signal' in self.lines:
                self.lines['Original Signal'].set_data(self.time_data, original_data)
        
        # Update frequency band signals
        if band_data:
            for band, data in band_data.items():
                if band in self.lines and len(data) > 0:
                    # Adjust data length
                    if len(data) > len(self.time_data):
                        data = data[:len(self.time_data)]
                    elif len(data) < len(self.time_data):
                        padded_data = np.zeros(len(self.time_data))
                        padded_data[:len(data)] = data
                        data = padded_data
                    
                    self.lines[band].set_data(self.time_data, data)
        
        # Dynamically adjust Y-axis range
        all_data = []
        for name, line in self.lines.items():
            if line.get_visible():
                x_data, y_data = line.get_data()
                if len(y_data) > 0:
                    all_data.extend(y_data)
        
        if all_data:
            y_min, y_max = min(all_data), max(all_data)
            margin = (y_max - y_min) * 0.1 + 0.1
            self.ax.set_ylim(y_min - margin, y_max + margin)
        else:
            self.ax.set_ylim(-200, 200)  # Adjusted to microvolt level range
        
        # Update legend
        self._update_legend()
        
        # Force redraw
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
    
    def clear_signals(self):
        """Clear signal display"""
        for line in self.lines.values():
            line.set_data([], [])
        self.ax.set_ylim(-200, 200)  # Adjusted to microvolt level range
        self.canvas.draw()


class AllChannelsEEGWidget(QWidget):
    """Widget to display all EEG channel waveforms with frequency band options"""
    
    def __init__(self):
        super().__init__()
        self.eeg_channels = []
        self.channel_axes = {}
        self.channel_lines = {}  # Dictionary: channel -> {signal_type: line}
        self.time_data = np.linspace(0, 5, 1250)  # 5 second window
        
        # Define signal colors
        self.colors = {
            'Original': 'black',
            'Delta': 'purple',
            'Theta': 'blue', 
            'Alpha': 'green',
            'Beta': 'orange',
            'Gamma': 'red'
        }
        
        # Default visibility settings
        self.visible_signals = {
            'Original': True,
            'Delta': False,
            'Theta': False,
            'Alpha': False,
            'Beta': False,
            'Gamma': False
        }
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        
        # Create control panel for frequency band selection
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # Create scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)  # Width adapts to window
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create content widget
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(5, 5, 5, 5)  # Small margins
        self.content_layout.setSpacing(0)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(12, 8))  # Initial size
        self.canvas = FigureCanvas(self.figure)
        
        # Set canvas size policy: horizontal expand, vertical fixed
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        self.content_layout.addWidget(self.canvas)
        self.content_widget.setLayout(self.content_layout)
        self.scroll_area.setWidget(self.content_widget)
        
        layout.addWidget(self.scroll_area)
        self.setLayout(layout)
    
    def create_control_panel(self):
        """Create control panel for frequency band selection"""
        panel = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title label
        title_label = QLabel("Display:")
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title_label)
        
        # Add checkboxes
        self.checkboxes = {}
        signal_names = ['Original', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        
        for name in signal_names:
            checkbox = QCheckBox(name)
            checkbox.setChecked(self.visible_signals.get(name, False))
            checkbox.stateChanged.connect(lambda state, n=name: self.on_checkbox_changed(n, state))
            
            # Set checkbox color
            if name in self.colors:
                checkbox.setStyleSheet(f"QCheckBox {{ color: {self.colors[name]}; font-weight: bold; }}")
            
            self.checkboxes[name] = checkbox
            layout.addWidget(checkbox)
        
        # Add stretch space
        layout.addStretch()
        
        # Add select all/deselect all buttons
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all)
        self.select_all_btn.setMaximumWidth(100)
        layout.addWidget(self.select_all_btn)
        
        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.clicked.connect(self.deselect_all)
        self.deselect_all_btn.setMaximumWidth(100)
        layout.addWidget(self.deselect_all_btn)
        
        panel.setLayout(layout)
        return panel
    
    def on_checkbox_changed(self, signal_name, state):
        """Checkbox state changed"""
        self.visible_signals[signal_name] = (state == Qt.Checked)
        self._update_line_visibility()
    
    def select_all(self):
        """Select all signals"""
        for name in self.checkboxes:
            self.checkboxes[name].setChecked(True)
            self.visible_signals[name] = True
        self._update_line_visibility()
    
    def deselect_all(self):
        """Deselect all signals"""
        for name in self.checkboxes:
            self.checkboxes[name].setChecked(False)
            self.visible_signals[name] = False
        self._update_line_visibility()
    
    def _update_line_visibility(self):
        """Update line visibility for all channels"""
        for channel in self.eeg_channels:
            if channel in self.channel_lines:
                for signal_type, line in self.channel_lines[channel].items():
                    visible = self.visible_signals.get(signal_type, False)
                    line.set_visible(visible)
        
        self.canvas.draw()
        
    def set_channels(self, channels):
        """Set the list of channels to display"""
        self.eeg_channels = channels
        self.channel_axes = {}
        self.channel_lines = {}
        
        # Clear old charts
        self.figure.clear()
        
        if len(channels) == 0:
            self.canvas.draw()
            return
        
        # Calculate appropriate chart height (allocate certain height per channel)
        height_per_channel = 1.5  # inches
        total_height = max(8, len(channels) * height_per_channel)
        
        # Adjust figure size
        self.figure.set_figheight(total_height)
        
        # Get DPI and calculate pixel height
        dpi = self.figure.get_dpi()
        pixel_height = int(total_height * dpi)
        
        # Create subplots, one per channel
        for idx, channel in enumerate(channels):
            ax = self.figure.add_subplot(len(channels), 1, idx + 1)
            
            # Clean channel name
            clean_name = channel.replace('EEG ', '').strip()
            
            ax.set_ylabel(clean_name, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 5)
            ax.set_ylim(-200, 200)
            
            # Show x-axis label only on the last subplot
            if idx == len(channels) - 1:
                ax.set_xlabel("Time (s)", fontsize=10)
            else:
                ax.set_xticklabels([])
            
            # Initialize multiple lines for each signal type
            self.channel_lines[channel] = {}
            for signal_type, color in self.colors.items():
                line, = ax.plot([], [], color=color, linewidth=1.0, 
                               alpha=0.8, label=signal_type,
                               visible=self.visible_signals.get(signal_type, False))
                self.channel_lines[channel][signal_type] = line
            
            self.channel_axes[channel] = ax
        
        # Adjust subplot spacing
        self.figure.subplots_adjust(hspace=0.1, left=0.1, right=0.95, top=0.98, bottom=0.05)
        
        # Set fixed canvas height so scroll bar works properly
        self.canvas.setFixedHeight(pixel_height)
        
        # Draw chart
        self.canvas.draw()
        
        # Print debug info
        print(f"Set up {len(channels)} channels, chart height: {total_height:.1f} inches ({pixel_height} pixels)")
    
    def update_signals(self, channel_data_dict, channel_band_data_dict=None):
        """
        Update signal display for all channels
        
        Args:
            channel_data_dict: Dictionary with channel names as keys and original signal data as values
            channel_band_data_dict: Dictionary with channel names as keys and frequency band data dict as values
                                   Format: {channel: {'Delta': data, 'Theta': data, ...}}
        """
        if len(self.eeg_channels) == 0:
            return
        
        for channel in self.eeg_channels:
            if channel not in channel_data_dict:
                continue
            
            original_data = channel_data_dict[channel]
            
            if len(original_data) == 0:
                continue
            
            # Adjust data length
            if len(original_data) > len(self.time_data):
                original_data = original_data[:len(self.time_data)]
            elif len(original_data) < len(self.time_data):
                padded_data = np.zeros(len(self.time_data))
                padded_data[:len(original_data)] = original_data
                original_data = padded_data
            
            # Update original signal line
            if channel in self.channel_lines and 'Original' in self.channel_lines[channel]:
                self.channel_lines[channel]['Original'].set_data(self.time_data, original_data)
            
            # Update frequency band lines if available
            if channel_band_data_dict and channel in channel_band_data_dict:
                band_data = channel_band_data_dict[channel]
                
                for band_name, band_signal in band_data.items():
                    if len(band_signal) == 0:
                        continue
                    
                    # Adjust band data length
                    if len(band_signal) > len(self.time_data):
                        band_signal = band_signal[:len(self.time_data)]
                    elif len(band_signal) < len(self.time_data):
                        padded_data = np.zeros(len(self.time_data))
                        padded_data[:len(band_signal)] = band_signal
                        band_signal = padded_data
                    
                    # Update band line
                    if band_name in self.channel_lines[channel]:
                        self.channel_lines[channel][band_name].set_data(self.time_data, band_signal)
            
            # Dynamically adjust Y-axis range based on visible lines
            if channel in self.channel_lines:
                all_visible_data = []
                for signal_type, line in self.channel_lines[channel].items():
                    if line.get_visible():
                        x_data, y_data = line.get_data()
                        if len(y_data) > 0:
                            all_visible_data.extend(y_data)
                
                if all_visible_data:
                    y_min, y_max = min(all_visible_data), max(all_visible_data)
                    margin = (y_max - y_min) * 0.1 + 10  # Add 10uV base margin
                    self.channel_axes[channel].set_ylim(y_min - margin, y_max + margin)
        
        # Redraw canvas
        self.canvas.draw()
    
    def clear_signals(self):
        """Clear all signal displays"""
        for channel in self.eeg_channels:
            if channel in self.channel_lines:
                for line in self.channel_lines[channel].values():
                    line.set_data([], [])
        
        for ax in self.channel_axes.values():
            ax.set_ylim(-200, 200)
        
        self.canvas.draw()


class AttentionTrendWidget(QWidget):
    """Attention trend display widget"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
        # Historical data
        self.score_history = []
        self.time_history = []  # Store time points for each score
        self.max_history = 72000  # Store up to 2 hour of history (72000 points at 100ms interval)
        self.update_interval = 100  # Default update interval in ms
        self.current_time = 0  # Current accumulated time
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        
        # Trend plot
        self.trend_plot = MatplotlibPlotWidget(
            title="Attention Score Trend (Full History)",
            ylabel="Score",
            xlabel="Elapsed Time (s)"
        )
        self.trend_plot.ax.set_ylim(0, 100)
        self.trend_plot.ax.set_facecolor('white')
        self.trend_plot.figure.patch.set_facecolor('white')
        
        # Format X-axis to show time clearly
        self.trend_plot.ax.grid(True, alpha=0.3, which='both')
        
        layout.addWidget(self.trend_plot)
        self.setLayout(layout)
    
    def update_trend(self, score):
        """Update trend plot"""
        # Accumulate time
        time_increment = self.update_interval / 1000.0  # Convert ms to seconds
        self.current_time += time_increment
        
        # Store score and corresponding time
        self.score_history.append(score)
        self.time_history.append(self.current_time)
        
        # Keep only recent history to prevent memory issues
        if len(self.score_history) > self.max_history:
            self.score_history.pop(0)
            self.time_history.pop(0)
        
        if len(self.score_history) > 1:
            # Use stored time history for X-axis data
            x_data = np.array(self.time_history)
            y_data = np.array(self.score_history)
            self.trend_plot.line.set_data(x_data, y_data)
            
            # Always show full history from 0 to current time
            # Use the actual first time point instead of 0 (in case old data was removed)
            x_min = self.time_history[0] if self.time_history else 0
            x_max = max(self.current_time, x_min + 5)  # At least 5 seconds range
            self.trend_plot.ax.set_xlim(x_min, x_max)
            
            # Update X-axis label with current time info
            if self.current_time >= 60:
                # Show minutes when time exceeds 60 seconds
                minutes = int(self.current_time // 60)
                seconds = int(self.current_time % 60)
                data_points = len(self.score_history)
                self.trend_plot.ax.set_xlabel(f"Elapsed Time (s) - Total: {minutes}m {seconds}s ({data_points} points)")
            else:
                data_points = len(self.score_history)
                self.trend_plot.ax.set_xlabel(f"Elapsed Time (s) - Total: {self.current_time:.1f}s ({data_points} points)")
            
            self.trend_plot.canvas.draw()
    
    def set_update_interval(self, interval_ms):
        """Set update interval in milliseconds"""
        self.update_interval = interval_ms
    
    def clear_trend(self):
        """Clear trend plot"""
        self.score_history.clear()
        self.time_history.clear()
        self.trend_plot.line.set_data([], [])
        # Reset accumulated time
        self.current_time = 0
        # Reset X-axis
        self.trend_plot.ax.set_xlim(0, 5)
        # Reset X-axis label
        self.trend_plot.ax.set_xlabel("Elapsed Time (s)")
        self.trend_plot.canvas.draw()


class SimpleMainWindow(QMainWindow):
    """Simplified main window"""
    
    def __init__(self):
        super().__init__()
        self.eeg_processor = EEGProcessor()
        self.is_file_mode = False
        self.file_position = 0
        self.eeg_channels = []  # Store detected EEG channels
        
        self.init_ui()
        self.setup_timer()
    
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("Real-time EEG Attention State Analysis System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set styles
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F5F5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #CCCCCC;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:pressed {
                background-color: #21618C;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left control panel
        control_panel = self.create_control_panel()
        
        # Right display area
        display_area = self.create_display_area()
        
        # Add to main layout
        main_layout.addWidget(control_panel, 1)
        main_layout.addWidget(display_area, 7)
        
        central_widget.setLayout(main_layout)
    
    def create_control_panel(self):
        """Create control panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # File operations group
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout()
        
        self.load_file_btn = QPushButton("Load EDF File")
        self.load_file_btn.clicked.connect(self.load_edf_file)
        file_layout.addWidget(self.load_file_btn)
        
        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # EEG channels information group
        channel_group = QGroupBox("EEG Channels Info")
        channel_layout = QVBoxLayout()
        
        # Channel count label
        self.channel_count_label = QLabel("Detected: 0 EEG channels")
        self.channel_count_label.setStyleSheet("font-weight: bold; color: #2E86AB; font-size: 15px;")
        channel_layout.addWidget(self.channel_count_label)
        
        # Channel list display (scrollable)
        self.channel_list_text = QTextEdit()
        self.channel_list_text.setReadOnly(True)
        self.channel_list_text.setMaximumHeight(100)
        self.channel_list_text.setStyleSheet("background-color: #F8F8F8; font-size: 15px;")
        self.channel_list_text.setPlaceholderText("No EEG channels loaded...")
        channel_layout.addWidget(self.channel_list_text)
        
        channel_info = QLabel("All EEG channels are automatically used for analysis\nEach channel can display original signal and frequency bands")
        channel_info.setWordWrap(True)
        channel_info.setStyleSheet("color: #666; font-size: 11px;")
        channel_layout.addWidget(channel_info)
        
        channel_group.setLayout(channel_layout)
        layout.addWidget(channel_group)
        
        # Preprocessing options group
        preprocess_group = QGroupBox("Preprocessing Options")
        preprocess_layout = QVBoxLayout()
        
        # Enable/disable all preprocessing
        self.preprocess_enabled_cb = QCheckBox("Enable Preprocessing")
        self.preprocess_enabled_cb.setChecked(True)
        self.preprocess_enabled_cb.setStyleSheet("font-weight: bold; color: #2E86AB; font-size: 12px;")
        self.preprocess_enabled_cb.stateChanged.connect(self.on_preprocessing_toggle)
        preprocess_layout.addWidget(self.preprocess_enabled_cb)
        
        # Individual preprocessing options
        self.bandpass_cb = QCheckBox("Bandpass Filter (0.5-100 Hz)")
        self.bandpass_cb.setChecked(True)
        self.bandpass_cb.setStyleSheet("font-size: 11px;")
        preprocess_layout.addWidget(self.bandpass_cb)
        
        self.notch_cb = QCheckBox("Notch Filter (50/60 Hz)")
        self.notch_cb.setChecked(True)
        self.notch_cb.setStyleSheet("font-size: 11px;")
        preprocess_layout.addWidget(self.notch_cb)
        
        self.baseline_cb = QCheckBox("Baseline Drift Removal")
        self.baseline_cb.setChecked(True)
        self.baseline_cb.setStyleSheet("font-size: 11px;")
        preprocess_layout.addWidget(self.baseline_cb)
        
        self.reref_cb = QCheckBox("Average Re-reference")
        self.reref_cb.setChecked(True)
        self.reref_cb.setStyleSheet("font-size: 11px;")
        preprocess_layout.addWidget(self.reref_cb)
        
        self.artifact_cb = QCheckBox("Artifact Detection & Removal")
        self.artifact_cb.setChecked(True)
        self.artifact_cb.setStyleSheet("font-size: 11px;")
        preprocess_layout.addWidget(self.artifact_cb)
        
        preprocess_info = QLabel("Note: Changes take effect on next file load")
        preprocess_info.setWordWrap(True)
        preprocess_info.setStyleSheet("color: #E67E22; font-size: 10px; font-style: italic;")
        preprocess_layout.addWidget(preprocess_info)
        
        preprocess_group.setLayout(preprocess_layout)
        layout.addWidget(preprocess_group)
        
        # Playback control group
        control_group = QGroupBox("Playback Control")
        control_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("Start Analysis")
        self.start_btn.clicked.connect(self.start_file_mode)
        self.start_btn.setEnabled(False)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_analysis)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Attention score display
        score_group = QGroupBox("Attention Score")
        score_layout = QVBoxLayout()
        
        self.score_label = QLabel("0")
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setFont(QFont("Arial", 36, QFont.Bold))
        self.score_label.setStyleSheet("""
            QLabel {
                color: #2E86AB;
                background-color: #F8F8F8;
                border: 2px solid #2E86AB;
                border-radius: 10px;
                padding: 20px;
            }
        """)
        
        self.status_label = QLabel("Status: Unknown")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Arial", 12))
        
        score_layout.addWidget(self.score_label)
        score_layout.addWidget(self.status_label)
        score_group.setLayout(score_layout)
        layout.addWidget(score_group)
        
        # Attention calculation mode selection
        mode_group = QGroupBox("Calculation Mode")
        mode_layout = QVBoxLayout()
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Relative Power",
            "Logarithmic Power",
            "Beta/Theta Ratio",
            "Frontal-Selective (Physiological)"
        ])
        self.mode_combo.setCurrentIndex(0)  # Default: Relative Power
        self.mode_combo.currentIndexChanged.connect(self.change_attention_mode)
        
        mode_info_label = QLabel("Select attention scoring algorithm")
        mode_info_label.setWordWrap(True)
        mode_info_label.setStyleSheet("color: #666; font-size: 10px;")
        
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addWidget(mode_info_label)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Update speed control
        speed_group = QGroupBox("Update Speed")
        speed_layout = QVBoxLayout()
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(50, 1000)
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(self.update_speed)
        
        self.speed_label = QLabel("100ms")
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_label)
        
        speed_group.setLayout(speed_layout)
        layout.addWidget(speed_group)
        
        # System information
        info_group = QGroupBox("System Info")
        info_layout = QVBoxLayout()
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(100)
        info_layout.addWidget(self.info_text)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Add stretch space
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def create_display_area(self):
        """Create display area"""
        # Use splitter
        splitter = QSplitter(Qt.Vertical)
        
        # All channels EEG signal display (new full channel display)
        self.all_channels_widget = AllChannelsEEGWidget()
        splitter.addWidget(self.all_channels_widget)
        
        # Attention trend display
        self.trend_widget = AttentionTrendWidget()
        splitter.addWidget(self.trend_widget)
        
        # Set splitter proportions (give more space to EEG signals)
        splitter.setSizes([600, 200])
        
        return splitter
    
    def setup_timer(self):
        """Setup timer"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.update_interval = 100  # Default 100ms
        
        # Initialize trend widget's update interval
        self.trend_widget.set_update_interval(self.update_interval)
    
    def load_edf_file(self):
        """Load EDF file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select EDF File", "", "EDF Files (*.edf);;All Files (*)"
        )
        
        if file_path:
            # Apply preprocessing options before loading
            self.apply_preprocessing_options()
            
            if self.eeg_processor.load_edf_file(file_path):
                self.file_label.setText(f"Loaded: {file_path.split('/')[-1]}")
                self.update_channel_list()
                self.start_btn.setEnabled(True)
                self.add_info_message(f"Successfully loaded file: {file_path}")
            else:
                self.file_label.setText("Load failed")
                self.add_info_message(f"Failed to load file: {file_path}", "error")
    
    def update_channel_list(self):
        """Update channel list and display EEG channels info"""
        if self.eeg_processor.channels:
            # Filter EEG channels
            self.eeg_channels = [ch for ch in self.eeg_processor.channels if 'EEG' in ch.upper()]
            
            # Update count label
            self.channel_count_label.setText(f"Detected: {len(self.eeg_channels)} EEG channels")
            
            # Display channel list in a formatted way
            if self.eeg_channels:
                # Format channels in a grid-like display (4 columns)
                channel_display = []
                for i in range(0, len(self.eeg_channels), 4):
                    row = self.eeg_channels[i:i+4]
                    # Clean channel names (remove 'EEG ' prefix if exists)
                    clean_row = [ch.replace('EEG ', '').strip() for ch in row]
                    channel_display.append('  '.join(f"{ch:<12}" for ch in clean_row))
                
                self.channel_list_text.setText('\n'.join(channel_display))
                
                # Set channel list for new widget
                self.all_channels_widget.set_channels(self.eeg_channels)
            else:
                self.channel_list_text.setText("No EEG channels found in this file")
                
            self.add_info_message(f"Detected {len(self.eeg_channels)} EEG channels")
    
    def change_attention_mode(self, index):
        """Change attention calculation mode"""
        mode_map = {
            0: 'relative',          # Relative Power
            1: 'log',               # Logarithmic Power
            2: 'ratio',             # Beta/Theta Ratio
            3: 'frontal_selective'  # Frontal-Selective (Physiological)
        }
        
        mode_names = {
            0: 'Relative Power',
            1: 'Logarithmic Power',
            2: 'Beta/Theta Ratio',
            3: 'Frontal-Selective (Physiological)'
        }
        
        mode = mode_map.get(index, 'relative')
        self.eeg_processor.set_attention_mode(mode)
        
        # Note: All modes now automatically process all EEG electrode channels
        # Channel selection is only used for waveform display
        self.add_info_message(f"Switched to {mode_names[index]} mode (auto-processes all EEG channels)")
    
    def on_preprocessing_toggle(self, state):
        """Handle preprocessing enable/disable toggle"""
        enabled = (state == Qt.Checked)
        
        # Enable/disable individual preprocessing checkboxes
        self.bandpass_cb.setEnabled(enabled)
        self.notch_cb.setEnabled(enabled)
        self.baseline_cb.setEnabled(enabled)
        self.reref_cb.setEnabled(enabled)
        self.artifact_cb.setEnabled(enabled)
        
        if enabled:
            self.add_info_message("Preprocessing enabled")
        else:
            self.add_info_message("Preprocessing disabled")
    
    def apply_preprocessing_options(self):
        """Apply preprocessing options to EEG processor"""
        # Get checkbox states
        enabled = self.preprocess_enabled_cb.isChecked()
        bandpass_enabled = self.bandpass_cb.isChecked()
        notch_enabled = self.notch_cb.isChecked()
        baseline_enabled = self.baseline_cb.isChecked()
        reref_enabled = self.reref_cb.isChecked()
        artifact_enabled = self.artifact_cb.isChecked()
        
        # Set main preprocessing flag
        self.eeg_processor.preprocess_enabled = enabled
        
        # Set individual preprocessing flags
        self.eeg_processor.bandpass_enabled = bandpass_enabled
        self.eeg_processor.notch_enabled = notch_enabled
        self.eeg_processor.baseline_enabled = baseline_enabled
        self.eeg_processor.reref_enabled = reref_enabled
        self.eeg_processor.artifact_enabled = artifact_enabled
        
        # Log current settings
        all_options = []
        
        if enabled:
            options = []
            if bandpass_enabled:
                options.append("Bandpass")
            if notch_enabled:
                options.append("Notch")
            if baseline_enabled:
                options.append("Baseline")
            if reref_enabled:
                options.append("Re-reference")
            if artifact_enabled:
                options.append("Artifact removal")
            
            all_options.extend(options)
        
        if all_options:
            self.add_info_message(f"Processing options: {', '.join(all_options)}")
        else:
            self.add_info_message("All preprocessing disabled")
    
    def start_file_mode(self):
        """Start file mode"""
        if not self.eeg_channels:
            self.add_info_message("No EEG channels detected in file", "warning")
            return
        
        self.is_file_mode = True
        self.file_position = 0
        self.timer.start(self.update_interval)
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        self.add_info_message(f"Started analysis (processing {len(self.eeg_channels)} EEG channels)")
    
    
    def stop_analysis(self):
        """Stop analysis"""
        self.timer.stop()
        
        self.is_file_mode = False
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # Clear display
        self.all_channels_widget.clear_signals()
        self.trend_widget.clear_trend()
        self.score_label.setText("0")
        self.status_label.setText("Status: Unknown")
        
        self.add_info_message("Stopped analysis")
    
    def update_display(self):
        """Update display"""
        if self.is_file_mode:
            # File mode - get data for all EEG channels
            channel_data_dict = self.get_all_channels_data(self.file_position, 5.0)
            
            if len(channel_data_dict) > 0:
                # Update file position
                self.file_position += 0.1  # Advance 0.1 seconds each time
                
                # Check if reached end of file
                max_time = self.eeg_processor.current_data.shape[1] / self.eeg_processor.sample_rate
                if self.file_position + 5.0 > max_time:
                    # Stop analysis when file ends instead of looping
                    self.add_info_message("File playback completed, analysis stopped")
                    self.stop_analysis()
                    return
                
                self.process_and_display(channel_data_dict)
            else:
                # No data available, stop analysis
                self.add_info_message("No data available, stopping analysis", "warning")
                self.stop_analysis()
                return
    
    def process_and_display(self, channel_data_dict):
        """Process and display data"""
        if len(channel_data_dict) == 0:
            return
        
        # Extract frequency bands for each channel
        channel_band_data_dict = {}
        for channel, original_data in channel_data_dict.items():
            if len(original_data) > 0:
                # Extract frequency bands from original signal
                band_data = self.eeg_processor.extract_frequency_bands(original_data)
                channel_band_data_dict[channel] = band_data
        
        # Update EEG signal display for all channels (with frequency bands)
        self.all_channels_widget.update_signals(channel_data_dict, channel_band_data_dict)
        
        # Calculate attention score
        # Now automatically processes all EEG electrode channels
        start_time = self.file_position if hasattr(self, 'file_position') else 0.0
        attention_score = self.eeg_processor.calculate_attention_score(
            start_time=start_time,
            duration=5.0  # Match the data window duration
        )
        
        # Update score display
        self.score_label.setText(f"{attention_score:.1f}")
        
        # Update status description
        if attention_score >= 80:
            status = "Status: Highly Focused"
            color = "#27AE60"
        elif attention_score >= 60:
            status = "Status: Good"
            color = "#F39C12"
        elif attention_score >= 40:
            status = "Status: Fair"
            color = "#E67E22"
        else:
            status = "Status: Unfocused"
            color = "#E74C3C"
        
        self.status_label.setText(status)
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        
        # Update trend plot
        self.trend_widget.update_trend(attention_score)
    
    def update_speed(self, value):
        """Update display speed"""
        self.update_interval = value
        self.speed_label.setText(f"{value}ms")
        
        # Update trend widget's update interval
        self.trend_widget.set_update_interval(value)
        
        if self.timer.isActive():
            self.timer.stop()
            self.timer.start(self.update_interval)
    
    def get_all_channels_data(self, start_time, duration):
        """
        Get data for all EEG channels
        
        Args:
            start_time: Start time (seconds)
            duration: Duration (seconds)
            
        Returns:
            dict: Dictionary with channel names as keys and signal data as values
        """
        if not self.eeg_channels or self.eeg_processor.current_data is None:
            return {}
        
        # Collect data from all EEG channels
        channel_data_dict = {}
        for channel in self.eeg_channels:
            data = self.eeg_processor.get_channel_data(channel, start_time, duration)
            if len(data) > 0:
                channel_data_dict[channel] = data
        
        return channel_data_dict
    
    def add_info_message(self, message, msg_type="info"):
        """Add info message"""
        color_map = {
            "info": "black",
            "warning": "orange", 
            "error": "red"
        }
        
        color = color_map.get(msg_type, "black")
        self.info_text.append(f'<span style="color: {color}">{message}</span>')
        
        # Scroll to bottom
        cursor = self.info_text.textCursor()
        cursor.movePosition(cursor.End)
        self.info_text.setTextCursor(cursor)
    
    def closeEvent(self, event):
        """Window close event"""
        self.stop_analysis()
        event.accept()


def main():
    """Main function"""
    app = QApplication(sys.argv)
    
    # Set application attributes
    app.setApplicationName("EEG Attention Analysis System")
    app.setApplicationVersion("1.0")
    
    # Create and show main window
    window = SimpleMainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
