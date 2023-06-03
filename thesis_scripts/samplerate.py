import matplotlib.pyplot as plt
import numpy as np


t = np.arange(0, 10, 0.1, dtype=float)
s = 3 * np.sin(t)
s2 = 3 * np.sin(t, dtype=float)
s2 = np.round(s2, decimals=0)
s2 = np.array(s2, dtype=int)
plt.xlim(0, 5)
plt.plot(t, s, style="-", color="red")
plt.plot(t, s2, color="blue")
plt.xlabel("Czas")
plt.ylabel("Amplituda")
plt.legend(["oryginalny", "po kwantyzacji"])
plt.show()
