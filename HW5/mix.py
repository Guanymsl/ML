import pandas as pd
import matplotlib.pyplot as plt

file_path_10 = './10.csv'
file_path_11 = './11.csv'
file_path_12 = './12.csv'

df_10 = pd.read_csv(file_path_10)
df_11 = pd.read_csv(file_path_11)
df_12 = pd.read_csv(file_path_12)

plt.figure(figsize=(10, 6))
plt.hist(df_10['E_out'], bins=30, color='orange', alpha=0.5, label=f'$E_{{in}}$')
plt.hist(df_11['E_out'], bins=30, color='green', alpha=0.5, label=f'$E_{{val}}$')
plt.title(f'Histogram of $E_{{out}}$')
plt.xlabel('$E_{{out}}$')
plt.ylabel('Frequency')
plt.legend()

plt.figure(figsize=(10, 6))
plt.hist(df_11['E_out'], bins=30, color='green', alpha=0.5, label=f'$E_{{val}}$')
plt.hist(df_12['E_out'], bins=30, color='purple', alpha=0.5, label=f'$E_{{cv}}$')
plt.title(f'Histogram of $E_{{out}}$')
plt.xlabel('$E_{{out}}$')
plt.ylabel('Frequency')
plt.legend()
plt.show()
