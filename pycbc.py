# Install PyCBC first if not installed
# pip install pycbc

import numpy as np
import matplotlib.pyplot as plt
from pycbc import waveform, noise, psd, types, matchedfilter

# 1️⃣ Basic GW150914 parameters (can modify later for other events)
mass1 = 36  # solar masses
mass2 = 29  # solar masses
f_low = 20  # Hz

# Generate original inspiral-merger-ringdown (IMR) waveform
hp, _ = waveform.get_td_waveform(approximant="SEOBNRv4",
                                  mass1=mass1,
                                  mass2=mass2,
                                  delta_t=1.0/4096,
                                  f_lower=f_low)

# Trim to same length always
hp = hp[:8192]
tlen = len(hp)

# 2️⃣ Create synthetic echo model (based on your toy model)
def generate_echoes(primary_waveform, delay_time=0.1, num_echoes=5, reflection_coeff=0.3):
    echo_train = primary_waveform.copy()
    dt = primary_waveform.delta_t
    for n in range(1, num_echoes+1):
        delay_samples = int(n * delay_time / dt)
        delayed_echo = reflection_coeff**n * primary_waveform.copy()
        delayed_echo = types.TimeSeries(np.pad(delayed_echo.numpy(), (delay_samples, 0), 'constant')[:len(primary_waveform)],
                                        delta_t=dt)
        echo_train += delayed_echo
    return echo_train

echo_waveform = generate_echoes(hp, delay_time=0.1, num_echoes=5, reflection_coeff=0.3)

# 3️⃣ Generate simulated Gaussian LIGO-like noise
psd_est = psd.aLIGOZeroDetHighPower(tlen//2 + 1, 1.0/hp.duration, f_low)
sim_noise = noise.noise_from_psd(len(hp), hp.delta_t, psd_est, seed=42)

# Inject the signal into noise
injected_signal = echo_waveform + sim_noise

# 4️⃣ Matched filtering to recover echo template
template = echo_waveform
snr = matchedfilter.matched_filter(template, injected_signal, psd=psd_est, low_frequency_cutoff=f_low)

# 5️⃣ Find peak SNR
peak = abs(snr).numpy().argmax()
peak_snr = abs(snr[peak])
peak_time = snr.sample_times[peak]

print(f"Recovered Peak SNR: {peak_snr:.2f} at time {peak_time:.3f} s")

# 6️⃣ Plot the injected signal & match-filter output
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(injected_signal.sample_times, injected_signal, label="Injected Signal (Echo + Noise)")
plt.xlabel("Time (s)")
plt.legend()

plt.subplot(1,2,2)
plt.plot(snr.sample_times, abs(snr), label="Matched Filter SNR")
plt.xlabel("Time (s)")
plt.legend()
plt.tight_layout()
plt.show()
