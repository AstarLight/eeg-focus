"""
Debug script to analyze why log method always gives ~100 score
"""

import numpy as np
from eeg_processor import EEGProcessor

# Initialize processor
processor = EEGProcessor(sample_rate=250)

# Test with typical EEG power values (μV²)
# Welch PSD integration typically gives values in range 1e-5 to 1e2
test_powers = [
    {'name': 'Typical Low', 'powers': {'Delta': 0.001, 'Theta': 0.0005, 'Alpha': 0.002, 'Beta': 0.003, 'Gamma': 0.0008}},
    {'name': 'Typical Medium', 'powers': {'Delta': 0.1, 'Theta': 0.05, 'Alpha': 0.2, 'Beta': 0.3, 'Gamma': 0.08}},
    {'name': 'Typical High', 'powers': {'Delta': 1.0, 'Theta': 0.5, 'Alpha': 2.0, 'Beta': 3.0, 'Gamma': 0.8}},
    {'name': 'Very High', 'powers': {'Delta': 10.0, 'Theta': 5.0, 'Alpha': 20.0, 'Beta': 30.0, 'Gamma': 8.0}},
]

print("="*80)
print("DEBUGGING LOG METHOD - Why scores are always ~100?")
print("="*80)

epsilon = 1e-10

for test_case in test_powers:
    print(f"\n{'-'*80}")
    print(f"Test Case: {test_case['name']}")
    print('-'*80)
    
    powers = test_case['powers']
    
    # Show raw powers
    print("\nRaw Band Powers (uV^2):")
    for band, power in powers.items():
        print(f"  {band:8s}: {power:.6f}")
    
    # Calculate log powers
    print("\nLog-transformed Powers:")
    log_score = 0.0
    for band_name, power in powers.items():
        log_power = np.log(power + epsilon)
        weight = processor.attention_weights[band_name]
        weighted = log_power * weight
        log_score += weighted
        print(f"  {band_name:8s}: log({power:.6f}) = {log_power:8.3f} × {weight:+.2f} = {weighted:+8.3f}")
    
    print(f"\nTotal log_score: {log_score:.3f}")
    
    # Calculate sigmoid
    sigmoid_input = -0.5 * (log_score + 10)
    print(f"Sigmoid input: -0.5 × ({log_score:.3f} + 10) = {sigmoid_input:.3f}")
    
    exp_value = np.exp(sigmoid_input)
    print(f"exp({sigmoid_input:.3f}) = {exp_value:.6f}")
    
    score = 100 / (1 + exp_value)
    print(f"\nFinal score: 100 / (1 + {exp_value:.6f}) = {score:.2f}/100")
    print(f"{'='*80}")

print("\n" + "="*80)
print("PROBLEM ANALYSIS")
print("="*80)
print("""
The issue is in the Sigmoid mapping parameters!

Current formula:
    score = 100 / (1 + exp(-0.5 * (log_score + 10)))

For typical EEG power values (0.001 to 10 uV^2):
    log(0.001) ~ -6.9
    log(0.01)  ~ -4.6
    log(0.1)   ~ -2.3
    log(1.0)   =  0.0
    log(10)    ~  2.3

After weighted sum with 5 bands, log_score is typically in range [-20, +10]

With the current offset (+10), the sigmoid input becomes:
    -0.5 * (log_score + 10)

When log_score = -20: sigmoid_input = -0.5 * (-10) = +5 -> score ~ 0.7%
When log_score = -10: sigmoid_input = -0.5 * (0)   = 0  -> score = 50%
When log_score = 0:   sigmoid_input = -0.5 * (10)  = -5 -> score ~ 99.3%
When log_score = +5:  sigmoid_input = -0.5 * (15)  = -7.5 -> score ~ 99.9%

PROBLEM: For typical EEG data where log_score > -10, the score is always >50
         and often >90!

The offset of +10 is TOO LARGE for the actual range of log_score values!
""")

print("\nSuggested fixes:")
print("1. Remove or reduce the offset (e.g., +10 → 0 or +5)")
print("2. Adjust the sigmoid steepness parameter (e.g., 0.5 → 1.0)")
print("3. Use dynamic normalization based on actual power ranges")

