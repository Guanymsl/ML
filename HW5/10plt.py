import pandas as pd
import matplotlib.pyplot as plt

file_path = './10.csv'
df = pd.read_csv(file_path)

plt.figure(figsize=(10, 6))
plt.hist(df['E_out'], bins=30, color='blue')
plt.title(f'Histogram of $E_{{out}}$')
plt.xlabel('$E_{{out}}$')
plt.ylabel('Frequency')

plt.figure(figsize=(10, 6))
plt.hist(df['Non_zero_components'], bins=30, color='red')
plt.title(f'Histogram of Non-zero Components')
plt.xlabel('Non-zero Components')
plt.ylabel('Frequency')
plt.show()
