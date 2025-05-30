
import numpy as np
import matplotlib.pyplot as plt

# Time grid
t = np.linspace(0, 0.5, 1000)

# Quantum hair phase modulation (fractal-like)
np.random.seed(0)
phase_jitter = np.cumsum(np.random.normal(0, 0.05, len(t)))  # Random walk
quantum_hair_signal = np.cos(2 * np.pi * 50 * t + phase_jitter)

# ECO signal (smooth)
eco_signal = np.cos(2 * np.pi * 50 * t + 0.2 * np.sin(2 * np.pi * 5 * t))  # Low-frequency smooth modulation

# Plotting
plt.figure(figsize=(12,6))
plt.plot(t, quantum_hair_signal, label='Quantum Hair Echo (Fractal Modulation)', color='blue')
plt.plot(t, eco_signal, label='ECO Echo (Smooth Modulation)', color='orange')
plt.xlabel("Time (s)")
plt.ylabel("Strain (arbitrary units)")
plt.title("Phase Modulation Comparison: Quantum Hair vs. ECO Echoes")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
