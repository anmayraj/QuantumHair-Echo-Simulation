# Python code to simulate echo waveform injected into LIGO-like noise
import numpy as np
import matplotlib.pyplot as plt

# Time grid
t = np.linspace(0, 0.5, 10000)  # 0 to 0.5 sec, 10k points

# Primary ringdown signal (Gaussian-modulated sine)
primary_pulse = np.exp(-2000*(t - 0.05)**2) * np.cos(1500*np.pi*(t - 0.05))

# Echo parameters
delay_time = 0.035  # Echo delay in seconds
num_echoes = 5
reflection_coeff = 0.3

# Generate echo train
echo_train = primary_pulse.copy()
for n in range(1, num_echoes + 1):
    delayed = np.pad(primary_pulse * reflection_coeff**n, (int(n * delay_time * len(t) / t[-1]), 0), 'constant')[:len(t)]
    echo_train += delayed

# Add simulated Gaussian noise
np.random.seed(42)
noise = np.random.normal(0, 0.05, len(t))
simulated_data = echo_train + noise

# Plot the waveform
plt.figure(figsize=(12,6))
plt.plot(t, simulated_data, label='Simulated LIGO Data (Echoes + Noise)', color='blue')
plt.plot(t, echo_train, label='Echo Signal (Model Prediction)', color='orange', alpha=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Strain (arbitrary units)")
plt.title("Gravitational Wave Echo Simulation: Quantum Hair Model")
plt.legend()
plt.tight_layout()
plt.grid(alpha=0.3)
plt.show()
