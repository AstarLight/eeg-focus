"""
Debug EDF file, check signal units and data range
"""

import numpy as np
import mne

def analyze_edf_file(file_path):
    """Analyze detailed information of EDF file"""
    print(f"Analyzing file: {file_path}")
    print("=" * 50)
    
    try:
        # Load EDF file
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        
        print(f"Number of channels: {len(raw.ch_names)}")
        print(f"Sampling rate: {raw.info['sfreq']} Hz")
        print(f"Data duration: {raw.times[-1]:.2f} seconds")
        print(f"Number of data points: {raw.times.shape[0]}")
        
        print("\nChannel information:")
        print("-" * 30)
        for i, ch_name in enumerate(raw.ch_names):
            print(f"Channel {i}: {ch_name}")
            
        # Get data
        data = raw.get_data()
        print(f"\nData shape: {data.shape}")
        
        # Analyze statistics of each channel
        print("\nChannel signal statistics:")
        print("-" * 50)
        for i, ch_name in enumerate(raw.ch_names):
            channel_data = data[i, :]
            print(f"\nChannel {i} ({ch_name}):")
            print(f"  Min value: {np.min(channel_data):.6f}")
            print(f"  Max value: {np.max(channel_data):.6f}")
            print(f"  Mean: {np.mean(channel_data):.6f}")
            print(f"  Std dev: {np.std(channel_data):.6f}")
            print(f"  RMS: {np.sqrt(np.mean(channel_data**2)):.6f}")
            
            # Check if all zeros
            if np.all(channel_data == 0):
                print(f"  ⚠️  Warning: All data in this channel is 0!")
            else:
                # Check data range
                non_zero_data = channel_data[channel_data != 0]
                if len(non_zero_data) > 0:
                    print(f"  Non-zero data range: {np.min(non_zero_data):.6f} to {np.max(non_zero_data):.6f}")
                    print(f"  Non-zero data mean: {np.mean(non_zero_data):.6f}")
        
        # Check physical units and digital range in EDF header
        print(f"\nEDF file header information:")
        print("-" * 30)
        
        # Get physical units and digital range information
        if hasattr(raw, '_orig_chs'):
            for i, ch in enumerate(raw._orig_chs):
                if i < len(raw.ch_names):
                    print(f"Channel {raw.ch_names[i]}:")
                    print(f"  Physical unit: {ch['unit']}")
                    print(f"  Physical min: {ch['cal'] * ch['range'][0] + ch['offset']:.6f}")
                    print(f"  Physical max: {ch['cal'] * ch['range'][1] + ch['offset']:.6f}")
                    print(f"  Digital min: {ch['range'][0]}")
                    print(f"  Digital max: {ch['range'][1]}")
                    print(f"  Calibration factor: {ch['cal']:.6f}")
                    print(f"  Offset: {ch['offset']:.6f}")
        
        # Check first few data points
        print(f"\nFirst 10 data points example (first 3 channels):")
        print("-" * 40)
        for i in range(min(3, len(raw.ch_names))):
            print(f"Channel {raw.ch_names[i]}: {data[i, :10]}")
            
        return raw, data
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        return None, None

if __name__ == "__main__":
    # Analyze EDF file
    file_path = "Subject01_1.edf"
    raw, data = analyze_edf_file(file_path)
    
    if raw is not None:
        print(f"\nSummary:")
        print("=" * 30)
        
        # Check if there is non-zero data
        has_non_zero = False
        for i in range(len(raw.ch_names)):
            if not np.all(data[i, :] == 0):
                has_non_zero = True
                break
        
        if has_non_zero:
            print("✓ File contains non-zero signal data")
        else:
            print("✗ All channel data in file is 0")
            
            # Try different loading methods
            print("\nTrying other loading methods...")
            try:
                # Try without preload
                raw2 = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
                data2 = raw2.get_data()
                print(f"Non-preload mode - Data shape: {data2.shape}")
                print(f"Non-preload mode - First 10 data points: {data2[0, :10] if len(data2) > 0 else 'No data'}")
            except Exception as e:
                print(f"Other loading methods also failed: {e}")
