"""
EEG Signal Processing Module
Responsible for loading, preprocessing, frequency band analysis and attention scoring calculation of EEG signals
"""

import numpy as np
import mne
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Optional, Tuple, Dict, List


class EEGProcessor:
    """EEG Signal Processor"""
    
    def __init__(self, sample_rate: int = 250):
        """
        Initialize EEG Processor
        
        Args:
            sample_rate: Sampling rate, default 250Hz
        """
        self.sample_rate = sample_rate
        self.raw_data = None
        self.current_data = None
        self.channels = None
        
        # Frequency band definitions (Hz)
        self.frequency_bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 45)
        }
        
        # Attention scoring weights (based on neuroscience research)
        self.attention_weights = {
            'Beta': 0.6,      # Beta waves positively correlated with attention focus
            'Alpha': -0.5,    # Alpha waves related to relaxation state, high Alpha may indicate lack of attention
            'Theta': -0.3,    # Theta waves related to drowsiness and meditation state
            'Delta': -0.1,    # Delta waves related to deep sleep
            'Gamma': 0.2      # Gamma waves related to higher cognitive functions
        }
        
        # Attention score calculation mode
        # 'relative': Relative power method
        # 'log': Logarithmic power method
        # 'ratio': Beta/Theta ratio method
        # 'frontal_selective': Frontal Beta + Non-frontal Alpha method (based on selective attention physiology)
        self.attention_mode = 'relative'
        
        # Define brain regions for frontal_selective mode
        # Frontal regions (task-relevant): Beta increase indicates focused attention
        self.frontal_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz']
        # Non-frontal regions (task-irrelevant): Alpha increase indicates selective inhibition
        # Converge to parietal-occipital sites for selective inhibition robustness
        self.non_frontal_channels = ['P3', 'P4', 'Pz', 'O1', 'O2']
        
        # Preprocessing parameters
        self.preprocess_enabled = True  # Enable preprocessing by default
        self.resample_enabled = True  # Enable resampling to target sample rate
        self.bandpass_enabled = True  # Enable bandpass filter
        self.notch_enabled = True  # Enable notch filter
        self.baseline_enabled = True  # Enable baseline drift removal
        self.reref_enabled = True  # Enable re-referencing
        self.artifact_enabled = True  # Enable artifact detection and handling
        self.bandpass_freq = (0.5, 100)  # Bandpass filter frequency range
        self.notch_freqs = [50, 60]  # Notch filter frequencies (power line interference)
        self.artifact_threshold = 150  # Artifact detection threshold (μV)
        self.bad_channels = []  # List to store bad channels detected
    
    def load_edf_file(self, file_path: str) -> bool:
        """
        Load EEG file in EDF format with preprocessing
        
        Args:
            file_path: Path to EDF file
            
        Returns:
            bool: Whether loading was successful
        """
        try:
            # Load EDF file using MNE
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False, encoding='latin1')
            self.raw_data = raw
            self.channels = raw.ch_names
            
            original_sfreq = raw.info['sfreq']
            print(f"Original sampling rate: {original_sfreq}Hz")
            
            # Resample to target sampling rate (if enabled)
            if self.resample_enabled and original_sfreq != self.sample_rate:
                print(f"Resampling from {original_sfreq}Hz to {self.sample_rate}Hz...")
                raw.resample(self.sample_rate, npad="auto")
                print(f"Resampling completed")
            elif not self.resample_enabled:
                print(f"Resampling disabled, keeping original rate: {original_sfreq}Hz")
                # Update sample_rate to match the actual data
                self.sample_rate = int(original_sfreq)
            else:
                print(f"Sample rate already matches target: {self.sample_rate}Hz")
            
            # Get data matrix (channels x times)
            data = raw.get_data()
            
            # Check data units and convert to microvolts
            # EDF files are typically in volts, need to convert to microvolts (multiply by 1e6)
            data_uv = self.convert_to_microvolts(data, raw)
            
            print(f"Successfully loaded EDF file: {file_path}")
            print(f"Number of channels: {len(self.channels)}")
            print(f"Number of samples: {data.shape[1]}")
            print(f"Sampling rate: {self.sample_rate}Hz")
            print(f"Data units converted to microvolts (μV)")
            
            # Display data range information before preprocessing
            if len(data_uv) > 0:
                sample_data = data_uv[0, :]  # First channel as sample
                print(f"Before preprocessing - Sample data range: {np.min(sample_data):.2f} to {np.max(sample_data):.2f} μV")
                print(f"Before preprocessing - Sample data mean: {np.mean(sample_data):.2f} μV")
                print(f"Before preprocessing - Sample data std dev: {np.std(sample_data):.2f} μV")
            
            # Apply preprocessing if enabled
            if self.preprocess_enabled:
                print("\n=== Starting EEG Preprocessing ===")
                data_uv = self.preprocess_eeg_data(data_uv)
                print("=== Preprocessing Complete ===\n")
                
                # Display data range information after preprocessing
                if len(data_uv) > 0:
                    sample_data = data_uv[0, :]  # First channel as sample
                    print(f"After preprocessing - Sample data range: {np.min(sample_data):.2f} to {np.max(sample_data):.2f} μV")
                    print(f"After preprocessing - Sample data mean: {np.mean(sample_data):.2f} μV")
                    print(f"After preprocessing - Sample data std dev: {np.std(sample_data):.2f} μV")
            
            self.current_data = data_uv
            
            return True
            
        except Exception as e:
            print(f"Failed to load EDF file: {e}")
            return False
    
    def convert_to_microvolts(self, data: np.ndarray, raw) -> np.ndarray:
        """
        Convert data to microvolt units
        
        Args:
            data: Raw data (possibly in volts)
            raw: MNE Raw object containing unit information
            
        Returns:
            np.ndarray: Data converted to microvolts
        """
        try:
            # Check data range to determine if unit conversion is needed
            data_max = np.max(np.abs(data))
            
            # If data range is very small (less than 1V), it may be in volts and needs conversion to microvolts
            if data_max < 1.0:
                print(f"Detected data in volts, converting to microvolts...")
                print(f"Data range before conversion: {-data_max:.6f} to {data_max:.6f} V")
                
                # Convert to microvolts (multiply by 1e6)
                data_uv = data * 1e6
                
                print(f"Data range after conversion: {np.min(data_uv):.2f} to {np.max(data_uv):.2f} μV")
                return data_uv
            else:
                # If data range is large, it may already be in microvolts
                print(f"Data range is {data_max:.2f}, may already be in microvolts")
                return data
                
        except Exception as e:
            print(f"Unit conversion failed: {e}")
            print(f"Using raw data, may affect display")
            return data
    
    def preprocess_eeg_data(self, data: np.ndarray) -> np.ndarray:
        """
        Comprehensive EEG data preprocessing pipeline
        
        Steps (conditionally applied based on settings):
        1. Bandpass filtering (0.5-100 Hz)
        2. Notch filtering (50Hz and 60Hz power line interference)
        3. Baseline drift removal
        4. Re-referencing (average reference)
        5. Artifact detection and handling
        
        Args:
            data: Input data (channels x samples) in microvolts
            
        Returns:
            np.ndarray: Preprocessed data
        """
        step_num = 1
        
        if self.bandpass_enabled:
            print(f"{step_num}. Applying bandpass filter: {self.bandpass_freq[0]}-{self.bandpass_freq[1]} Hz...")
            data = self.apply_bandpass_filter(data, self.bandpass_freq[0], self.bandpass_freq[1])
            step_num += 1
        else:
            print(f"{step_num}. Bandpass filter: Skipped")
            step_num += 1
        
        if self.notch_enabled:
            print(f"{step_num}. Applying notch filters at {self.notch_freqs} Hz (power line interference)...")
            for freq in self.notch_freqs:
                data = self.apply_notch_filter(data, freq)
            step_num += 1
        else:
            print(f"{step_num}. Notch filter: Skipped")
            step_num += 1
        
        if self.baseline_enabled:
            print(f"{step_num}. Removing baseline drift...")
            data = self.remove_baseline_drift(data)
            step_num += 1
        else:
            print(f"{step_num}. Baseline drift removal: Skipped")
            step_num += 1
        
        if self.reref_enabled:
            print(f"{step_num}. Applying average re-reference...")
            data = self.apply_average_reference(data)
            step_num += 1
        else:
            print(f"{step_num}. Re-reference: Skipped")
            step_num += 1
        
        if self.artifact_enabled:
            print(f"{step_num}. Detecting and handling artifacts...")
            data, n_artifacts = self.detect_and_handle_artifacts(data)
            print(f"   Detected and interpolated {n_artifacts} artifact segments")
        else:
            print(f"{step_num}. Artifact detection: Skipped")
        
        return data
    
    def apply_bandpass_filter(self, data: np.ndarray, low_freq: float, high_freq: float) -> np.ndarray:
        """
        Apply bandpass filter to all channels
        
        Args:
            data: Input data (channels x samples)
            low_freq: Low frequency cutoff (Hz)
            high_freq: High frequency cutoff (Hz)
            
        Returns:
            np.ndarray: Filtered data
        """
        filtered_data = np.zeros_like(data)
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Design Butterworth bandpass filter
        b, a = signal.butter(4, [low, high], btype='band')
        
        for i in range(data.shape[0]):
            filtered_data[i, :] = signal.filtfilt(b, a, data[i, :])
        
        return filtered_data
    
    def apply_notch_filter(self, data: np.ndarray, notch_freq: float, quality_factor: float = 30.0) -> np.ndarray:
        """
        Apply notch filter to remove power line interference
        
        Args:
            data: Input data (channels x samples)
            notch_freq: Frequency to notch out (Hz)
            quality_factor: Quality factor (higher = narrower notch)
            
        Returns:
            np.ndarray: Filtered data
        """
        filtered_data = np.zeros_like(data)
        
        # Design notch filter
        b, a = signal.iirnotch(notch_freq, quality_factor, self.sample_rate)
        
        for i in range(data.shape[0]):
            filtered_data[i, :] = signal.filtfilt(b, a, data[i, :])
        
        return filtered_data
    
    def remove_baseline_drift(self, data: np.ndarray, window_size: Optional[int] = None) -> np.ndarray:
        """
        Remove baseline drift using high-pass filtering or detrending
        
        Args:
            data: Input data (channels x samples)
            window_size: Window size for moving average (if None, uses simple detrending)
            
        Returns:
            np.ndarray: Data with baseline drift removed
        """
        corrected_data = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            # Use polynomial detrending (removes linear and polynomial trends)
            corrected_data[i, :] = signal.detrend(data[i, :], type='linear')
        
        return corrected_data
    
    def apply_average_reference(self, data: np.ndarray) -> np.ndarray:
        """
        Apply average reference to EEG data
        Re-references all channels to the average of all channels
        
        Args:
            data: Input data (channels x samples)
            
        Returns:
            np.ndarray: Re-referenced data
        """
        # Calculate average across all channels
        avg_reference = np.mean(data, axis=0)
        
        # Subtract average from each channel
        referenced_data = data - avg_reference
        
        return referenced_data
    
    def detect_and_handle_artifacts(self, data: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Detect and handle artifacts using amplitude threshold method
        
        Artifacts are detected when amplitude exceeds threshold.
        Simple interpolation is used to handle detected artifacts.
        
        Args:
            data: Input data (channels x samples)
            
        Returns:
            Tuple[np.ndarray, int]: (Cleaned data, Number of artifacts detected)
        """
        cleaned_data = data.copy()
        n_artifacts = 0
        
        for i in range(data.shape[0]):
            channel_data = data[i, :]
            
            # Detect artifacts (amplitude exceeding threshold)
            artifact_mask = np.abs(channel_data) > self.artifact_threshold
            
            if np.any(artifact_mask):
                # Count artifacts
                n_artifacts += np.sum(artifact_mask)
                
                # Simple interpolation: replace artifact points with interpolated values
                artifact_indices = np.where(artifact_mask)[0]
                good_indices = np.where(~artifact_mask)[0]
                
                if len(good_indices) > 1:
                    # Interpolate using good data points
                    cleaned_data[i, artifact_indices] = np.interp(
                        artifact_indices, 
                        good_indices, 
                        channel_data[good_indices]
                    )
        
        return cleaned_data, n_artifacts
    
    def set_preprocessing_params(self, 
                                   enabled: bool = True,
                                   bandpass_freq: Optional[Tuple[float, float]] = None,
                                   notch_freqs: Optional[List[float]] = None,
                                   artifact_threshold: Optional[float] = None):
        """
        Set preprocessing parameters
        
        Args:
            enabled: Enable/disable preprocessing
            bandpass_freq: Bandpass filter frequency range (low, high) in Hz
            notch_freqs: List of notch filter frequencies in Hz
            artifact_threshold: Artifact detection threshold in μV
        """
        self.preprocess_enabled = enabled
        
        if bandpass_freq is not None:
            self.bandpass_freq = bandpass_freq
            
        if notch_freqs is not None:
            self.notch_freqs = notch_freqs
            
        if artifact_threshold is not None:
            self.artifact_threshold = artifact_threshold
        
        print(f"Preprocessing parameters updated:")
        print(f"  Enabled: {self.preprocess_enabled}")
        print(f"  Bandpass: {self.bandpass_freq} Hz")
        print(f"  Notch filters: {self.notch_freqs} Hz")
        print(f"  Artifact threshold: {self.artifact_threshold} μV")
    
    def get_channel_data(self, channel_name: str, start_time: float = 0, duration: float = 2) -> np.ndarray:
        """
        Get data segment from specified channel
        
        Args:
            channel_name: Channel name
            start_time: Start time (seconds)
            duration: Duration (seconds)
            
        Returns:
            np.ndarray: Signal data
        """
        if self.current_data is None:
            return np.array([])
        
        if channel_name not in self.channels:
            return np.array([])
        
        channel_idx = self.channels.index(channel_name)
        start_sample = int(start_time * self.sample_rate)
        end_sample = int((start_time + duration) * self.sample_rate)
        
        # Ensure index doesn't go out of bounds
        end_sample = min(end_sample, self.current_data.shape[1])
        
        return self.current_data[channel_idx, start_sample:end_sample]
    
    def bandpass_filter(self, data: np.ndarray, low_freq: float, high_freq: float) -> np.ndarray:
        """
        Bandpass filter
        
        Args:
            data: Input signal
            low_freq: Low frequency cutoff
            high_freq: High frequency cutoff
            
        Returns:
            np.ndarray: Filtered signal
        """
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Design Butterworth bandpass filter
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter
        filtered_data = signal.filtfilt(b, a, data)
        
        return filtered_data
    
    def extract_frequency_bands(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract signals from each frequency band
        
        Args:
            data: Input signal
            
        Returns:
            Dict[str, np.ndarray]: Signals from each frequency band
        """
        band_data = {}
        
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            band_data[band_name] = self.bandpass_filter(data, low_freq, high_freq)
        
        return band_data
    
    def calculate_band_power(self, data: np.ndarray, band_name: str) -> float:
        """
        Calculate frequency band power using Welch's method
        
        Args:
            data: Input signal
            band_name: Frequency band name
            
        Returns:
            float: Band power (μV²)
        """
        if band_name not in self.frequency_bands:
            return 0.0
        
        low_freq, high_freq = self.frequency_bands[band_name]
        
        # Use Welch's method to calculate power spectral density (PSD)
        # nperseg: Prefer 2-second windows for stable Alpha/Beta estimates; 50% overlap
        nperseg = min(len(data), int(2 * self.sample_rate))
        noverlap = max(0, nperseg // 2)
        freqs, psd = signal.welch(data, fs=self.sample_rate, nperseg=nperseg, noverlap=noverlap)
        
        # Find indices corresponding to frequency band
        idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
        
        # Calculate band power by integrating PSD in the frequency band
        # Use trapezoidal rule for numerical integration
        freq_res = freqs[1] - freqs[0]  # Frequency resolution
        band_power = np.trapz(psd[idx_band], dx=freq_res)
        
        return band_power
    
    def set_attention_mode(self, mode: str) -> bool:
        """
        Set attention score calculation mode
        
        Args:
            mode: Calculation mode - 'relative', 'log', 'ratio', or 'frontal_selective'
        
        Returns:
            bool: True if successfully set, False if invalid mode
        """
        valid_modes = ['relative', 'log', 'ratio', 'frontal_selective']
        if mode not in valid_modes:
            print(f"Error: Invalid mode '{mode}'. Valid modes: {valid_modes}")
            return False
        
        self.attention_mode = mode
        print(f"Attention calculation mode set to: {mode}")
        return True
    
    def get_attention_mode(self) -> str:
        """Get current attention calculation mode"""
        return self.attention_mode
    
    def calculate_attention_score(self, start_time: float = 0, duration: float = 2.0) -> float:
        """
        Calculate attention score using selected mode
        Automatically processes all EEG electrode channels
        
        Modes:
        - 'relative': Relative power method (original)
        - 'log': Logarithmic power method
        - 'ratio': Beta/Theta ratio method (classic neuroscience indicator)
        - 'frontal_selective': Frontal Beta + Non-frontal Alpha (selective attention physiology)
        
        Args:
            start_time: Start time for analysis (seconds)
            duration: Duration for analysis (seconds, default 2.0)
            
        Returns:
            float: Attention score (0-100)
        """
        if self.current_data is None:
            return 0.0
        
        # Select calculation method based on mode
        if self.attention_mode == 'relative':
            return self._calculate_score_relative(start_time=start_time, duration=duration)
        elif self.attention_mode == 'log':
            return self._calculate_score_log(start_time=start_time, duration=duration)
        elif self.attention_mode == 'ratio':
            return self._calculate_score_ratio(start_time=start_time, duration=duration)
        elif self.attention_mode == 'frontal_selective':
            return self._calculate_score_frontal_selective(start_time=start_time, duration=duration)
        else:
            return self._calculate_score_relative(start_time=start_time, duration=duration)
    
    def _calculate_score_relative(self, start_time: float = 0, duration: float = 2.0) -> float:
        """
        Method 1: Relative power method
        Uses relative power ratios weighted by attention weights
        Processes all EEG electrode channels
        
        Args:
            start_time: Start time for analysis (seconds)
            duration: Duration for analysis (seconds)
            
        Returns:
            float: Attention score (0-100)
        """
        if self.current_data is None or self.channels is None:
            return 0.0
        
        epsilon = 1e-10
        all_channel_scores = []
        
        # Iterate through all EEG channels
        for channel in self.channels:
            # Filter for EEG channels only
            if 'EEG' not in channel.upper():
                continue
            
            channel_data = self.get_channel_data(channel, start_time, duration)
            
            if len(channel_data) == 0:
                continue
            
            # Calculate band powers for this channel
            band_powers = {}
            for band_name in self.frequency_bands.keys():
                power = self.calculate_band_power(channel_data, band_name)
                band_powers[band_name] = power
            
            total_power = sum(band_powers.values())
            if total_power == 0:
                continue
            
            # Calculate weighted attention score for this channel
            attention_score = 0.0
            for band_name, power in band_powers.items():
                relative_power = power / total_power
                weighted_score = relative_power * self.attention_weights[band_name]
                attention_score += weighted_score
            
            all_channel_scores.append(attention_score)
        
        # Check if we have valid scores
        if len(all_channel_scores) == 0:
            return 0.0
        
        # Average scores across all EEG channels
        avg_attention_score = np.mean(all_channel_scores)
        
        # Map to 0-100 range
        min_possible_score = min(self.attention_weights.values())
        max_possible_score = max(self.attention_weights.values())
        score_range = max_possible_score - min_possible_score
        
        # Linear mapping to [0, 1]
        if score_range > 0:
            normalized_score = (avg_attention_score - min_possible_score) / score_range
        else:
            normalized_score = 0.5
        
        score_0_100 = normalized_score * 100
        score_0_100 = max(0, min(100, score_0_100))
        
        return score_0_100
    
    def _calculate_score_log(self, start_time: float = 0, duration: float = 2.0) -> float:
        """
        Method 2: Logarithmic power method
        Uses log-transformed absolute power to preserve intensity information
        Processes all EEG electrode channels
        
        Args:
            start_time: Start time for analysis (seconds)
            duration: Duration for analysis (seconds)
            
        Returns:
            float: Attention score (0-100)
        """
        if self.current_data is None or self.channels is None:
            return 0.0
        
        epsilon = 1e-10
        all_channel_scores = []
        
        # Iterate through all EEG channels
        for channel in self.channels:
            # Filter for EEG channels only
            if 'EEG' not in channel.upper():
                continue
            
            channel_data = self.get_channel_data(channel, start_time, duration)
            
            if len(channel_data) == 0:
                continue
            
            # Calculate band powers for this channel
            band_powers = {}
            for band_name in self.frequency_bands.keys():
                power = self.calculate_band_power(channel_data, band_name)
                band_powers[band_name] = power
            
            # Calculate log-weighted score for this channel
            log_score = 0.0
            log_powers = {}
            
            for band_name, power in band_powers.items():
                log_power = np.log(power + epsilon)
                log_powers[band_name] = log_power
                log_score += log_power * self.attention_weights[band_name]
            
            # Dynamic normalization
            min_weight = min(self.attention_weights.values())
            max_weight = max(self.attention_weights.values())
            
            min_possible_score = min([log_power * min_weight for log_power in log_powers.values()])
            max_possible_score = max([log_power * max_weight for log_power in log_powers.values()])
            
            score_range = max_possible_score - min_possible_score
            
            # Linear mapping to [0, 1]
            if score_range > 1e-6:
                normalized_score = (log_score - min_possible_score) / score_range
            else:
                normalized_score = 1 / (1 + np.exp(-1.5 * log_score))
            
            all_channel_scores.append(normalized_score)
        
        # Check if we have valid scores
        if len(all_channel_scores) == 0:
            return 0.0
        
        # Average scores across all EEG channels
        avg_normalized_score = np.mean(all_channel_scores)
        score_0_100 = avg_normalized_score * 100
        score_0_100 = max(0, min(100, score_0_100))
        
        return score_0_100
    
    def _calculate_score_ratio(self, start_time: float = 0, duration: float = 2.0) -> float:
        """
        Method 3: Beta/Theta ratio method
        Classic neuroscience attention indicator
        High Beta/Theta ratio indicates high attention
        Processes all EEG electrode channels
        
        Args:
            start_time: Start time for analysis (seconds)
            duration: Duration for analysis (seconds)
            
        Returns:
            float: Attention score (0-100)
        """
        if self.current_data is None or self.channels is None:
            return 0.0
        
        epsilon = 1e-10
        all_channel_scores = []
        
        # Iterate through all EEG channels
        for channel in self.channels:
            # Filter for EEG channels only
            if 'EEG' not in channel.upper():
                continue
            
            channel_data = self.get_channel_data(channel, start_time, duration)
            
            if len(channel_data) == 0:
                continue
            
            # Calculate band powers for this channel
            beta_power = self.calculate_band_power(channel_data, 'Beta')
            alpha_power = self.calculate_band_power(channel_data, 'Alpha')
            theta_power = self.calculate_band_power(channel_data, 'Theta')
            
            # Calculate Beta/(Alpha + Theta) ratio
            ratio = beta_power / (alpha_power + theta_power + epsilon)
            
            # Also consider Beta/Theta ratio
            beta_theta_ratio = beta_power / (theta_power + epsilon)
            
            # Combine ratios with log transformation
            combined_ratio = np.log(ratio + epsilon) + 0.5 * np.log(beta_theta_ratio + epsilon)
            
            # Map to 0-100 using sigmoid
            score = 100 / (1 + np.exp(-0.8 * combined_ratio))
            all_channel_scores.append(score)
        
        # Check if we have valid scores
        if len(all_channel_scores) == 0:
            return 0.0
        
        # Average scores across all EEG channels
        score_0_100 = np.mean(all_channel_scores)
        score_0_100 = max(0, min(100, score_0_100))
        
        return score_0_100
    
    
    def _calculate_score_frontal_selective(self, start_time: float = 0, duration: float = 2) -> float:
        """
        Method 5: Frontal Beta + Non-frontal Alpha method
        Based on selective attention physiology:
        - Task-relevant regions (frontal): Beta increase indicates focused processing
        - Task-irrelevant regions (non-frontal): Alpha increase indicates selective inhibition
        
        Args:
            start_time: Start time for analysis (seconds)
            duration: Duration for analysis (seconds)
            
        Returns:
            float: Attention score (0-100)
        """
        if self.current_data is None or self.channels is None:
            return 0.0
        
        epsilon = 1e-10
        
        # Collect relative power from frontal, non-frontal regions, and all channels
        frontal_beta_relative = []
        non_frontal_alpha_relative = []
        all_beta_relative = []
        all_alpha_relative = []
        
        # Single iteration through all channels - optimized to avoid redundant calculations
        for channel in self.channels:
            clean_name = channel.replace('EEG ', '').strip()
            channel_data = self.get_channel_data(channel, start_time, duration)
            
            if len(channel_data) == 0:
                continue
            
            # Calculate all band powers once for this channel
            delta_power = self.calculate_band_power(channel_data, 'Delta')
            theta_power = self.calculate_band_power(channel_data, 'Theta')
            alpha_power = self.calculate_band_power(channel_data, 'Alpha')
            beta_power = self.calculate_band_power(channel_data, 'Beta')
            gamma_power = self.calculate_band_power(channel_data, 'Gamma')
            
            total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power + epsilon
            
            # Calculate relative powers
            beta_rel = beta_power / total_power
            alpha_rel = alpha_power / total_power
            
            # Collect data for all channels (for reference distribution)
            all_beta_relative.append(beta_rel)
            all_alpha_relative.append(alpha_rel)
            
            # Collect data for specific regions
            if clean_name in self.frontal_channels:
                frontal_beta_relative.append(beta_rel)
            
            if clean_name in self.non_frontal_channels:
                non_frontal_alpha_relative.append(alpha_rel)
        
        # Data validity check
        if len(frontal_beta_relative) == 0 or len(non_frontal_alpha_relative) == 0:
            return 0.0
        
        # Calculate mean and std for z-score normalization
        beta_mean = np.mean(all_beta_relative)
        beta_std = np.std(all_beta_relative) + epsilon
        alpha_mean = np.mean(all_alpha_relative)
        alpha_std = np.std(all_alpha_relative) + epsilon
        
        # Calculate average relative power for target regions
        avg_frontal_beta_rel = np.mean(frontal_beta_relative)
        avg_non_frontal_alpha_rel = np.mean(non_frontal_alpha_relative)
        
        # Z-score normalization: measures how much above/below average
        # Positive z-score = higher than average (good for attention indicators)
        beta_z_score = (avg_frontal_beta_rel - beta_mean) / beta_std
        alpha_z_score = (avg_non_frontal_alpha_rel - alpha_mean) / alpha_std
        
        # Combined score: 70% frontal Beta z-score + 30% non-frontal Alpha z-score
        # Z-scores typically range from -3 to +3, centered at 0
        raw_score = 0.7 * beta_z_score + 0.3 * alpha_z_score
        
        # Map to 0-100 range using Sigmoid function
        # Slope of 1.5: z-score of ±2 maps to ~95% and ~5%
        score_0_100 = 100.0 / (1.0 + np.exp(-1.5 * raw_score))
        score_0_100 = max(0.0, min(100.0, float(score_0_100)))
        
        return score_0_100
