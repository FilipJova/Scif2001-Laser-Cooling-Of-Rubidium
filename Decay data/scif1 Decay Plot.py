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

time = np.arange(len(data))   # Assuming each frame is 0.1 seconds apart

plt.figure(figsize=(10, 6))
plt.plot(time, data, 'o', label='Data', markersize=4)
plt.xlabel('Time (s)')
plt.ylabel('Peak Luminosity')
plt.title('Peak Luminosity vs Time')
plt.grid()
plt.legend()
plt.show()