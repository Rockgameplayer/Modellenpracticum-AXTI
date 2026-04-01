import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("Test Plot")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid(True)
plt.show()
