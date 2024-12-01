import matplotlib.pyplot as plt
import numpy as np

gamma_values = [0.01, 0.1, 1.0, 10.0, 100.0]
frequencies = [120, 8, 0, 0, 0]
log_G = np.log10(gamma_values)

plt.bar(log_G, frequencies, width=0.5, color="orange")
plt.xticks(log_G, gamma_values)

plt.xlabel(r'$\gamma$')
plt.ylabel('Frequency')

plt.show()
