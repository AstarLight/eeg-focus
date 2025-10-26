"""
Visualize the working principle of remove_baseline_drift function
Demonstrate the linear detrending process using scipy.signal.detrend
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Set font for better display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def create_sample_eeg_signal():
    """Create a simulated EEG signal with baseline drift"""
    # Time axis
    t = np.linspace(0, 8, 100)  # 8 seconds, 100 sampling points
    
    # Create signal components
    # 1. Baseline drift (linear trend)
    baseline_drift = 2.5 * t + 10  # Linear upward trend
    
    # 2. Real EEG signal (Alpha wave + noise)
    alpha_wave = 5 * np.sin(2 * np.pi * 10 * t)  # 10Hz Alpha wave
    noise = np.random.normal(0, 1, len(t))  # Random noise
    
    # 3. Combined signal
    original_signal = baseline_drift + alpha_wave + noise
    
    return t, original_signal, baseline_drift, alpha_wave

def demonstrate_detrend():
    """Demonstrate the detrending process"""
    # Create sample signal
    t, original_signal, baseline_drift, alpha_wave = create_sample_eeg_signal()
    
    # Use scipy.signal.detrend for linear detrending
    detrended_signal = signal.detrend(original_signal, type='linear')
    
    # Manually calculate fitted baseline (for visualization)
    # Fit line: y = a*x + b
    coeffs = np.polyfit(t, original_signal, 1)
    fitted_baseline = coeffs[0] * t + coeffs[1]
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('EEG Signal Baseline Correction (Linear Detrending) Demo', fontsize=16, fontweight='bold')
    
    # First subplot: Original signal and fitted baseline
    axes[0].plot(t, original_signal, 'b-', linewidth=2, label='Original Signal (with baseline drift)')
    axes[0].plot(t, fitted_baseline, 'r--', linewidth=2, label=f'Fitted Baseline (y={coeffs[0]:.2f}x+{coeffs[1]:.2f})')
    axes[0].plot(t, alpha_wave, 'g:', linewidth=1, alpha=0.7, label='True EEG Signal (Alpha wave)')
    axes[0].set_ylabel('Amplitude (μV)')
    axes[0].set_title('Step 1: Original Signal and Linear Fitting')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Second subplot: Baseline drift components
    axes[1].plot(t, baseline_drift, 'r-', linewidth=2, label='True Baseline Drift')
    axes[1].plot(t, fitted_baseline, 'r--', linewidth=2, label='Fitted Baseline')
    axes[1].fill_between(t, baseline_drift, fitted_baseline, alpha=0.3, color='red', label='Fitting Error')
    axes[1].set_ylabel('Amplitude (μV)')
    axes[1].set_title('Step 2: Baseline Drift Analysis')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Third subplot: Corrected signal
    axes[2].plot(t, detrended_signal, 'g-', linewidth=2, label='Corrected Signal')
    axes[2].plot(t, alpha_wave, 'g:', linewidth=1, alpha=0.7, label='True EEG Signal (Reference)')
    axes[2].axhline(y=0, color='k', linestyle='-', alpha=0.5, label='Zero Baseline')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Amplitude (μV)')
    axes[2].set_title('Step 3: Baseline Correction Result')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Adjust subplot spacing
    plt.tight_layout()
    
    # Add statistical information
    print("=== Baseline Correction Statistics ===")
    print(f"Original signal range: {np.min(original_signal):.2f} ~ {np.max(original_signal):.2f} μV")
    print(f"Fitted baseline range: {np.min(fitted_baseline):.2f} ~ {np.max(fitted_baseline):.2f} μV")
    print(f"Corrected signal range: {np.min(detrended_signal):.2f} ~ {np.max(detrended_signal):.2f} μV")
    print(f"Fitted slope: {coeffs[0]:.2f} μV/second")
    print(f"Fitted intercept: {coeffs[1]:.2f} μV")
    
    return fig

def demonstrate_detrend_algorithm():
    """Detailed demonstration of detrending algorithm steps"""
    # Create simple example
    t = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    original_signal = np.array([10, 12, 15, 18, 20, 22, 25, 28])
    
    # Manual implementation of linear detrending
    # Step 1: Linear fitting
    coeffs = np.polyfit(t, original_signal, 1)
    fitted_baseline = coeffs[0] * t + coeffs[1]
    
    # Step 2: Subtract fitted baseline
    detrended_signal = original_signal - fitted_baseline
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('Linear Detrending Algorithm Detailed Demo', fontsize=14, fontweight='bold')
    
    # First subplot: Original signal and fitted baseline
    axes[0].plot(t, original_signal, 'bo-', linewidth=2, markersize=8, label='Original Signal')
    axes[0].plot(t, fitted_baseline, 'r--', linewidth=2, label=f'Fitted Baseline (y={coeffs[0]:.1f}x+{coeffs[1]:.1f})')
    axes[0].set_ylabel('Amplitude (μV)')
    axes[0].set_title('Step 1: Linear Fitting')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(t)
    
    # Second subplot: Corrected signal
    axes[1].plot(t, detrended_signal, 'go-', linewidth=2, markersize=8, label='Corrected Signal')
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.5, label='Zero Baseline')
    axes[1].set_xlabel('Time Points')
    axes[1].set_ylabel('Amplitude (μV)')
    axes[1].set_title('Step 2: Subtract Fitted Baseline')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(t)
    
    plt.tight_layout()
    
    # Print detailed calculation process
    print("\n=== Detailed Calculation Process ===")
    print("Time points:", t)
    print("Original signal:", original_signal)
    print("Fitted baseline:", fitted_baseline)
    print("Corrected signal:", detrended_signal)
    print(f"\nFitting parameters: slope={coeffs[0]:.2f}, intercept={coeffs[1]:.2f}")
    
    return fig

if __name__ == "__main__":
    print("Generating EEG baseline correction visualization...")
    
    # Generate first figure: Complete demonstration
    fig1 = demonstrate_detrend()
    fig1.savefig('eeg_detrend_demo_en.png', dpi=300, bbox_inches='tight')
    print("Saved: eeg_detrend_demo_en.png")
    
    # Generate second figure: Algorithm detailed demonstration
    fig2 = demonstrate_detrend_algorithm()
    fig2.savefig('detrend_algorithm_demo_en.png', dpi=300, bbox_inches='tight')
    print("Saved: detrend_algorithm_demo_en.png")
    
    # Show figures
    plt.show()
    
    print("\nVisualization completed!")
