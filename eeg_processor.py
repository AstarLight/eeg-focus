"""
EEG Signal Processing Module
Responsible for loading, preprocessing, frequency band analysis and attention scoring calculation of EEG signals
"""

import numpy as np
import mne
from scipy import signal
from scipy.fft import fft, fftfreq
import threading
import time
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
        self.is_running = False
        self.simulation_thread = None
        
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
            'Alpha': -0.2,    # Alpha waves related to relaxation state, high Alpha may indicate lack of attention
            'Theta': -0.3,    # Theta waves related to drowsiness and meditation state
            'Delta': -0.1,    # Delta waves related to deep sleep
            'Gamma': 0.2      # Gamma waves related to higher cognitive functions
        }
        
        # Attention score calculation mode
        # 'relative': Relative power method
        # 'log': Logarithmic power method
        # 'ratio': Beta/Theta ratio method
        # 'combined': Combined method
        # 'frontal_selective': Frontal Beta + Non-frontal Alpha method (based on selective attention physiology)
        self.attention_mode = 'relative'
        
        # Define brain regions for frontal_selective mode
        # Frontal regions (task-relevant): Beta increase indicates focused attention
        self.frontal_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz']
        # Non-frontal regions (task-irrelevant): Alpha increase indicates selective inhibition
        # Converge to parietal-occipital sites for selective inhibition robustness
        self.non_frontal_channels = ['P3', 'P4', 'Pz', 'O1', 'O2']
    
    def load_edf_file(self, file_path: str) -> bool:
        """
        Load EEG file in EDF format
        
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
            
            # Resample to target sampling rate
            if raw.info['sfreq'] != self.sample_rate:
                raw.resample(self.sample_rate, npad="auto")
            
            # Get data matrix (channels x times)
            data = raw.get_data()
            
            # Check data units and convert to microvolts
            # EDF files are typically in volts, need to convert to microvolts (multiply by 1e6)
            data_uv = self.convert_to_microvolts(data, raw)
            self.current_data = data_uv
            
            print(f"Successfully loaded EDF file: {file_path}")
            print(f"Number of channels: {len(self.channels)}")
            print(f"Number of samples: {data.shape[1]}")
            print(f"Sampling rate: {self.sample_rate}Hz")
            print(f"Data units converted to microvolts (μV)")
            
            # Display data range information
            if len(data_uv) > 0:
                sample_data = data_uv[0, :]  # First channel as sample
                print(f"Sample data range: {np.min(sample_data):.2f} to {np.max(sample_data):.2f} μV")
                print(f"Sample data mean: {np.mean(sample_data):.2f} μV")
                print(f"Sample data std dev: {np.std(sample_data):.2f} μV")
            
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
            mode: Calculation mode - 'relative', 'log', 'ratio', 'combined', or 'frontal_selective'
        
        Returns:
            bool: True if successfully set, False if invalid mode
        """
        valid_modes = ['relative', 'log', 'ratio', 'combined', 'frontal_selective']
        if mode not in valid_modes:
            print(f"Error: Invalid mode '{mode}'. Valid modes: {valid_modes}")
            return False
        
        self.attention_mode = mode
        print(f"Attention calculation mode set to: {mode}")
        return True
    
    def get_attention_mode(self) -> str:
        """Get current attention calculation mode"""
        return self.attention_mode
    
    def calculate_attention_score(self, data: np.ndarray, channel_name: Optional[str] = None, 
                                  start_time: Optional[float] = None, duration: float = 2.0) -> float:
        """
        Calculate attention score using selected mode
        
        Modes:
        - 'relative': Relative power method (original)
        - 'log': Logarithmic power method
        - 'ratio': Beta/Theta ratio method (classic neuroscience indicator)
        - 'combined': Combined method using all approaches
        - 'frontal_selective': Frontal Beta + Non-frontal Alpha (selective attention physiology)
        
        Args:
            data: Input signal (single channel for most modes, or all channels for frontal_selective)
            channel_name: Channel name (required for frontal_selective mode)
            start_time: Start time for analysis (required for frontal_selective mode in file playback)
            duration: Duration for analysis (default 2.0 seconds)
            
        Returns:
            float: Attention score (0-100)
        """
        if len(data) == 0:
            return 0.0
        
        # For frontal_selective mode, use multi-channel analysis
        if self.attention_mode == 'frontal_selective':
            # Use provided start_time if available, otherwise use 0
            if start_time is None:
                start_time = 0.0
            return self._calculate_score_frontal_selective(start_time=start_time, duration=duration)
        
        # Calculate power of each frequency band for single channel
        band_powers = {}
        for band_name in self.frequency_bands.keys():
            power = self.calculate_band_power(data, band_name)
            band_powers[band_name] = power
        
        # Select calculation method based on mode
        if self.attention_mode == 'relative':
            return self._calculate_score_relative(band_powers)
        elif self.attention_mode == 'log':
            return self._calculate_score_log(band_powers)
        elif self.attention_mode == 'ratio':
            return self._calculate_score_ratio(band_powers)
        elif self.attention_mode == 'combined':
            return self._calculate_score_combined(band_powers)
        else:
            return self._calculate_score_relative(band_powers)
    
    def _calculate_score_relative(self, band_powers: Dict[str, float]) -> float:
        """
        Method 1: Relative power method
        Uses relative power ratios weighted by attention weights
        """
        total_power = sum(band_powers.values())
        if total_power == 0:
            return 0.0
        
        # Calculate weighted attention score
        attention_score = 0.0
        for band_name, power in band_powers.items():
            relative_power = power / total_power
            weighted_score = relative_power * self.attention_weights[band_name]
            attention_score += weighted_score
        
        # Map to 0-100 range
        # Dynamically calculate theoretical range based on current weights
        # Min score: when all power is in the most negative weight band
        # Max score: when all power is in the most positive weight band
        min_possible_score = min(self.attention_weights.values())
        max_possible_score = max(self.attention_weights.values())
        score_range = max_possible_score - min_possible_score
        
        # Linear mapping to [0, 1]
        if score_range > 0:
            normalized_score = (attention_score - min_possible_score) / score_range
        else:
            normalized_score = 0.5  # Fallback if all weights are equal
        
        score_0_100 = normalized_score * 100
        
        # Ensure score is within reasonable range
        score_0_100 = max(0, min(100, score_0_100))
        
        return score_0_100
    
    def _calculate_score_log(self, band_powers: Dict[str, float]) -> float:
        """
        Method 2: Logarithmic power method
        Uses log-transformed absolute power to preserve intensity information
        """
        epsilon = 1e-10
        
        # Calculate log-weighted score
        log_score = 0.0
        log_powers = {}
        
        for band_name, power in band_powers.items():
            log_power = np.log(power + epsilon)
            log_powers[band_name] = log_power
            log_score += log_power * self.attention_weights[band_name]
        
        # Dynamic normalization based on actual log power values
        # Calculate theoretical min/max based on actual band powers
        # Worst case: 100% power in band with most negative weight
        # Best case: 100% power in band with most positive weight
        
        min_weight = min(self.attention_weights.values())
        max_weight = max(self.attention_weights.values())
        
        # Calculate what scores would be if all power was in min or max weight band
        # Using actual log power values from each band
        min_possible_score = min([log_power * min_weight for log_power in log_powers.values()])
        max_possible_score = max([log_power * max_weight for log_power in log_powers.values()])
        
        # Expand range slightly to account for weighted combinations
        score_range = max_possible_score - min_possible_score
        
        # Linear mapping to [0, 1]
        if score_range > 1e-6:
            # Normalize based on distance from minimum
            normalized_score = (log_score - min_possible_score) / score_range
        else:
            # Fallback: use sigmoid with adjusted parameters for very uniform cases
            # Center around 0, moderate slope
            normalized_score = 1 / (1 + np.exp(-1.5 * log_score))
        
        score_0_100 = normalized_score * 100
        
        # Ensure score is within reasonable range
        score_0_100 = max(0, min(100, score_0_100))
        
        return score_0_100
    
    def _calculate_score_ratio(self, band_powers: Dict[str, float]) -> float:
        """
        Method 3: Beta/Theta ratio method
        Classic neuroscience attention indicator
        High Beta/Theta ratio indicates high attention
        """
        epsilon = 1e-10
        
        # Calculate Beta/(Alpha + Theta) ratio - common attention metric
        beta_power = band_powers['Beta']
        alpha_power = band_powers['Alpha']
        theta_power = band_powers['Theta']
        
        # Main ratio: Beta / (Alpha + Theta)
        ratio = beta_power / (alpha_power + theta_power + epsilon)
        
        # Also consider Beta/Theta ratio
        beta_theta_ratio = beta_power / (theta_power + epsilon)
        
        # Combine ratios with log transformation
        combined_ratio = np.log(ratio + epsilon) + 0.5 * np.log(beta_theta_ratio + epsilon)
        
        # Map to 0-100 using sigmoid
        # Adjusted for typical ratio ranges
        score_0_100 = 100 / (1 + np.exp(-0.8 * combined_ratio))
        
        return score_0_100
    
    def _calculate_score_combined(self, band_powers: Dict[str, float]) -> float:
        """
        Method 4: Combined method
        Integrates all three approaches for robust scoring
        """
        # Get scores from all methods
        score_relative = self._calculate_score_relative(band_powers)
        score_log = self._calculate_score_log(band_powers)
        score_ratio = self._calculate_score_ratio(band_powers)
        
        # Weighted combination
        # Relative: 40%, Log: 30%, Ratio: 30%
        combined_score = (
            0.4 * score_relative +
            0.3 * score_log +
            0.3 * score_ratio
        )
        
        # Ensure score is within range
        combined_score = max(0, min(100, combined_score))
        
        return combined_score
    
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
        
        # Helper: robust z-score using median and MAD
        def robust_z(values: np.ndarray) -> np.ndarray:
            if len(values) == 0:
                return np.array([])
            med = np.median(values)
            mad = np.median(np.abs(values - med))
            scale = (mad * 1.4826) + 1e-12
            return (values - med) / scale
        
        # Collect per-channel relative powers
        frontal_beta_rel = []
        frontal_alpha_rel = []
        frontal_gamma_rel = []
        post_alpha_rel = []
        theta_rel_all = []
        
        for channel in self.channels:
            clean_name = channel.replace('EEG ', '').strip()
            channel_data = self.get_channel_data(channel, start_time, duration)
            if len(channel_data) == 0:
                continue
            # Band powers
            delta_p = self.calculate_band_power(channel_data, 'Delta')
            theta_p = self.calculate_band_power(channel_data, 'Theta')
            alpha_p = self.calculate_band_power(channel_data, 'Alpha')
            beta_p = self.calculate_band_power(channel_data, 'Beta')
            gamma_p = self.calculate_band_power(channel_data, 'Gamma')
            total_p = max(delta_p + theta_p + alpha_p + beta_p + gamma_p, epsilon)
            # Relative powers
            delta_rel = delta_p / total_p
            theta_rel = theta_p / total_p
            alpha_rel = alpha_p / total_p
            beta_rel = beta_p / total_p
            gamma_rel = gamma_p / total_p
            theta_rel_all.append(theta_rel)
            if clean_name in self.frontal_channels:
                frontal_beta_rel.append(beta_rel)
                frontal_alpha_rel.append(alpha_rel)
                frontal_gamma_rel.append(gamma_rel)
            if clean_name in self.non_frontal_channels:
                post_alpha_rel.append(alpha_rel)
        
        # Data sufficiency checks
        if len(frontal_beta_rel) == 0 or len(post_alpha_rel) == 0:
            return 0.0
        
        # Step 1: group robust medians on raw relative powers
        beta_frontal_med = float(np.median(frontal_beta_rel)) if len(frontal_beta_rel) > 0 else 0.0
        alpha_frontal_med = float(np.median(frontal_alpha_rel)) if len(frontal_alpha_rel) > 0 else 0.0
        alpha_post_med = float(np.median(post_alpha_rel)) if len(post_alpha_rel) > 0 else 0.0
        theta_global_med = float(np.median(theta_rel_all)) if len(theta_rel_all) > 0 else 0.0

        # Step 2: build global reference distributions across all channels (relative powers)
        # Use all-channel vectors for robust z computation
        # If some arrays are empty (unlikely for theta_rel_all), fallback to small epsilon
        all_beta_rel = []
        all_alpha_rel = []
        for channel in self.channels:
            clean_name = channel.replace('EEG ', '').strip()
            data_ch = self.get_channel_data(channel, start_time, duration)
            if len(data_ch) == 0:
                continue
            d = self.calculate_band_power(data_ch, 'Delta')
            t = self.calculate_band_power(data_ch, 'Theta')
            a = self.calculate_band_power(data_ch, 'Alpha')
            b = self.calculate_band_power(data_ch, 'Beta')
            g = self.calculate_band_power(data_ch, 'Gamma')
            tot = max(d + t + a + b + g, epsilon)
            all_beta_rel.append(b / tot)
            all_alpha_rel.append(a / tot)
        all_beta_rel = np.array(all_beta_rel) if len(all_beta_rel) > 0 else np.array([epsilon])
        all_alpha_rel = np.array(all_alpha_rel) if len(all_alpha_rel) > 0 else np.array([epsilon])
        all_theta_rel = np.array(theta_rel_all) if len(theta_rel_all) > 0 else np.array([epsilon])

        # Step 3: compute robust z of group medians against global distributions
        def robust_z_scalar(x: float, ref: np.ndarray) -> float:
            med = np.median(ref)
            mad = np.median(np.abs(ref - med))
            scale = (mad * 1.4826) + 1e-12
            return float((x - med) / scale)

        z_beta_frontal_med = robust_z_scalar(beta_frontal_med, all_beta_rel)
        z_alpha_frontal_med = robust_z_scalar(alpha_frontal_med, all_alpha_rel)
        z_alpha_post_med = robust_z_scalar(alpha_post_med, all_alpha_rel)  # 与同一Alpha参考对比
        z_theta_global_med = robust_z_scalar(theta_global_med, all_theta_rel)
        
        # Selective inhibition: posterior alpha high minus frontal alpha contribution
        selective_alpha = z_alpha_post_med - 0.5 * z_alpha_frontal_med
        
        # Light artifact guard: penalize high frontal gamma (EMG)
        frontal_gamma_rel_med = float(np.median(frontal_gamma_rel)) if len(frontal_gamma_rel) > 0 else 0.0
        # Penalty ramps from 1.0 (<=0.15) down to 0.5 (>=0.35)
        if frontal_gamma_rel_med <= 0.15:
            emg_penalty = 1.0
        elif frontal_gamma_rel_med >= 0.35:
            emg_penalty = 0.5
        else:
            ratio = (frontal_gamma_rel_med - 0.15) / (0.35 - 0.15)
            emg_penalty = 1.0 - 0.5 * ratio
        
        # Combine indicators
        raw_score = 0.7 * emg_penalty * z_beta_frontal_med + 0.3 * selective_alpha - 0.4 * z_theta_global_med
        
        # Map to 0-100 using sigmoid centered at 0
        slope = 1.5
        score_0_100 = 100.0 / (1.0 + np.exp(-slope * raw_score))
        score_0_100 = max(0.0, min(100.0, float(score_0_100)))
        return score_0_100
    
    def generate_simulated_signal(self, duration: float = 2.0) -> np.ndarray:
        """
        Generate simulated EEG signal
        
        Args:
            duration: Signal duration (seconds)
            
        Returns:
            np.ndarray: Simulated EEG signal
        """
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples)
        
        # Generate simulated signal containing components from each frequency band
        signal_data = np.zeros(n_samples)
        
        # Add sine waves from each frequency band to simulate real EEG characteristics
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            # Choose random frequency within band range
            freq = np.random.uniform(low_freq, high_freq)
            # Random amplitude and phase
            amplitude = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2 * np.pi)
            # Add to total signal
            signal_data += amplitude * np.sin(2 * np.pi * freq * t + phase)
        
        # Add some noise to make signal more realistic
        noise = np.random.normal(0, 0.5, n_samples)
        signal_data += noise
        
        return signal_data
    
    def start_simulation(self, callback_func):
        """
        Start simulating real-time signal
        
        Args:
            callback_func: Callback function for passing newly generated signal data
        """
        if self.is_running:
            return
        
        self.is_running = True
        self.simulation_thread = threading.Thread(target=self._simulation_loop, args=(callback_func,))
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
    
    def stop_simulation(self):
        """Stop simulated signal"""
        self.is_running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=1.0)
    
    def _simulation_loop(self, callback_func):
        """
        Simulation signal loop
        
        Args:
            callback_func: Callback function
        """
        while self.is_running:
            # Generate 5 seconds of simulated signal
            signal_data = self.generate_simulated_signal(5.0)
            
            # Call callback function to pass data
            if callback_func:
                callback_func(signal_data)
            
            # Wait for a while to simulate real-time acquisition
            time.sleep(0.1)
