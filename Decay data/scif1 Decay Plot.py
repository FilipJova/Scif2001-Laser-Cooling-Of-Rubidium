from matplotlib.cm import binary
from matplotlib.cm import binary
import numpy as np
from  pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import csv

script_dir = Path(__file__).parent.absolute()
csv_path = script_dir / 'peak_luminosity(6f).csv'

data = np.loadtxt(csv_path, delimiter=',', usecols=(1,), skiprows=1)

csv_path = script_dir / 'peak_luminosity(6f)2.csv'

data = np.append(data, np.loadtxt(csv_path, delimiter=',', usecols=(1,), skiprows=1))
data = data[data > 1]

print(data)

time = np.arange(len(data)) *5  # Assuming each frame is 0.1 seconds apart

def exponential_decay(t, A, tau, C):
    return A * np.exp(-t / tau) + C

p0 = (np.max(data), 1, np.min(data))  # Initial guess for A, tau, C
popt, pcov = curve_fit(exponential_decay, time, data, p0=p0)

tau = popt[1]
tau_err = np.sqrt(np.diag(pcov))[1]  # Error propagation for tau

print(f"Fitted decay constant (tau): {tau:.3f} ± {tau_err:.3f} seconds")

plt.figure(figsize=(10, 6))
plt.scatter(time, data, label='Data', s=10)
plt.plot(time, exponential_decay(time, *popt), '-', label='Fitted Curve', linewidth=2, color='red')
plt.xlabel('Time (s)')
plt.ylabel('Peak Luminosity')
plt.title('Peak Luminosity vs Time')
plt.grid()
plt.legend()
plt.show()