"""
Simplified Main Window Module
Uses matplotlib instead of pyqtgraph to reduce dependencies
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox, 
                             QFileDialog, QGroupBox, QGridLayout, QSlider,
                             QTextEdit, QSplitter, QCheckBox)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
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
        self.current_channel = None
        self.is_file_mode = False
        self.file_position = 0
        self.simulation_data = None
        
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
        
        # Channel selection group
        channel_group = QGroupBox("Channel Selection")
        channel_layout = QVBoxLayout()
        
        self.channel_combo = QComboBox()
        self.channel_combo.currentTextChanged.connect(self.on_channel_changed)
        channel_layout.addWidget(self.channel_combo)
        
        channel_group.setLayout(channel_layout)
        layout.addWidget(channel_group)
        
        # Signal source selection group
        source_group = QGroupBox("Signal Source")
        source_layout = QVBoxLayout()
        
        self.file_mode_btn = QPushButton("File Mode")
        self.file_mode_btn.clicked.connect(self.start_file_mode)
        self.file_mode_btn.setEnabled(False)
        source_layout.addWidget(self.file_mode_btn)
        
        self.simulation_btn = QPushButton("Simulated Signal")
        self.simulation_btn.clicked.connect(self.toggle_simulation)
        source_layout.addWidget(self.simulation_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_analysis)
        self.stop_btn.setEnabled(False)
        source_layout.addWidget(self.stop_btn)
        
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)
        
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
            "Combined Method"
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
        
        # Combined signal display (original signal + frequency band signals)
        self.combined_signal_widget = CombinedSignalWidget()
        splitter.addWidget(self.combined_signal_widget)
        
        # Attention trend display
        self.trend_widget = AttentionTrendWidget()
        splitter.addWidget(self.trend_widget)
        
        # Set splitter proportions
        splitter.setSizes([400, 200])
        
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
            if self.eeg_processor.load_edf_file(file_path):
                self.file_label.setText(f"Loaded: {file_path.split('/')[-1]}")
                self.update_channel_list()
                self.file_mode_btn.setEnabled(True)
                self.add_info_message(f"Successfully loaded file: {file_path}")
            else:
                self.file_label.setText("Load failed")
                self.add_info_message(f"Failed to load file: {file_path}", "error")
    
    def update_channel_list(self):
        """Update channel list"""
        self.channel_combo.clear()
        if self.eeg_processor.channels:
            self.channel_combo.addItems(self.eeg_processor.channels)
            self.current_channel = self.eeg_processor.channels[0]
    
    def on_channel_changed(self, channel_name):
        """Channel selection changed"""
        self.current_channel = channel_name
        self.add_info_message(f"Selected channel: {channel_name}")
    
    def change_attention_mode(self, index):
        """Change attention calculation mode"""
        mode_map = {
            0: 'relative',      # Relative Power
            1: 'log',           # Logarithmic Power
            2: 'ratio',         # Beta/Theta Ratio
            3: 'combined'       # Combined Method
        }
        
        mode_names = {
            0: 'Relative Power',
            1: 'Logarithmic Power',
            2: 'Beta/Theta Ratio',
            3: 'Combined Method'
        }
        
        mode = mode_map.get(index, 'relative')
        self.eeg_processor.set_attention_mode(mode)
        self.add_info_message(f"Switched to {mode_names[index]} mode")
    
    def start_file_mode(self):
        """Start file mode"""
        if not self.current_channel:
            self.add_info_message("Please select a channel first", "warning")
            return
        
        self.is_file_mode = True
        self.file_position = 0
        self.timer.start(self.update_interval)
        
        self.file_mode_btn.setEnabled(False)
        self.simulation_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        self.add_info_message("Started file mode analysis")
    
    def toggle_simulation(self):
        """Toggle simulated signal"""
        if self.simulation_btn.text() == "Simulated Signal":
            self.start_simulation()
        else:
            self.stop_simulation()
    
    def start_simulation(self):
        """Start simulated signal"""
        self.eeg_processor.start_simulation(self.on_simulation_data)
        self.simulation_btn.setText("Stop Simulation")
        self.file_mode_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.timer.start(self.update_interval)
        
        self.add_info_message("Started simulated signal")
    
    def stop_simulation(self):
        """Stop simulated signal"""
        self.eeg_processor.stop_simulation()
        self.simulation_btn.setText("Simulated Signal")
        self.file_mode_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.timer.stop()
        
        self.add_info_message("Stopped simulated signal")
    
    def on_simulation_data(self, data):
        """Simulated data callback"""
        self.simulation_data = data
    
    def stop_analysis(self):
        """Stop analysis"""
        self.timer.stop()
        self.eeg_processor.stop_simulation()
        
        self.is_file_mode = False
        self.simulation_data = None
        
        self.file_mode_btn.setEnabled(True)
        self.simulation_btn.setEnabled(True)
        self.simulation_btn.setText("Simulated Signal")
        self.stop_btn.setEnabled(False)
        
        # Clear display
        self.combined_signal_widget.clear_signals()
        self.trend_widget.clear_trend()
        self.score_label.setText("0")
        self.status_label.setText("Status: Unknown")
        
        self.add_info_message("Stopped analysis")
    
    def update_display(self):
        """Update display"""
        if self.is_file_mode:
            # File mode
            data = self.eeg_processor.get_channel_data(
                self.current_channel, 
                self.file_position, 
                5.0  # 5 seconds of data
            )
            
            if len(data) > 0:
                # Update file position
                self.file_position += 0.1  # Advance 0.1 seconds each time
                
                # Check if reached end of file
                max_time = self.eeg_processor.current_data.shape[1] / self.eeg_processor.sample_rate
                if self.file_position + 5.0 > max_time:
                    self.file_position = 0  # Loop playback
                
                self.process_and_display(data)
        
        elif self.simulation_data is not None:
            # Simulation mode
            self.process_and_display(self.simulation_data)
    
    def process_and_display(self, data):
        """Process and display data"""
        if len(data) == 0:
            return
        
        # Extract frequency bands
        band_data = self.eeg_processor.extract_frequency_bands(data)
        
        # Update combined signal display (original signal + frequency band signals)
        self.combined_signal_widget.update_signals(data, band_data)
        
        # Calculate attention score
        attention_score = self.eeg_processor.calculate_attention_score(data)
        
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
