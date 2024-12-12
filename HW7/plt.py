import matplotlib.pyplot as plt

metrics = {}
with open('./data/adaboost_metrics.txt', 'r') as f:
    lines = f.readlines()
    for i in range(0, len(lines), 2):
        key = lines[i].strip().strip(':')
        values = list(map(float, lines[i + 1].strip().split()))
        metrics[key] = values

T = range(1, 501)

plt.figure(figsize=(10, 6))
plt.title(r'$E_{in}(g_t)$ and $\epsilon_t$ vs $t$')
plt.plot(T, metrics['E_in_g_t'], label=r'$E_{in}(g_t)$', color='blue')
plt.plot(T, metrics['epsilon_t'], label=r'$\epsilon_t$', color='red')
plt.xlabel(r'Iterations $(t)$')
plt.ylabel('Accuracy / Value')
plt.legend()

plt.figure(figsize=(10, 6))
plt.title(r'$E_{in}(G_t)$ and $E_{out}(G_t)$ vs $t$')
plt.plot(T, metrics['E_in_G_t'], label=r'$E_{in}(G_t)$', color='blue')
plt.plot(T, metrics['E_out_G_t'], label=r'$E_{out}(G_t)$', color='red')
plt.xlabel(r'Iterations $(t)$')
plt.ylabel('Accuracy')
plt.legend()

plt.figure(figsize=(10, 6))
plt.title(r'$U_t$ and $E_{in}(G_t)$ vs $t$')
plt.plot(T, metrics['U_t'], label=r'$U_t$', color='blue')
plt.plot(T, metrics['E_in_G_t'], label=r'$E_{in}(G_t)$', color='red')
plt.xlabel(r'Iterations $(t)$')
plt.ylabel('Accuracy / Value')
plt.legend()

plt.show()
