import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-150,150,1)
y = 1 + 0.6 * (x / np.abs(x)) / (1 + np.exp(-np.abs(x)/15 + 4.5));
#y = 1 / (1 + np.exp(-x + 5))
plt.plot(x,y)
plt.title("double sigmoid function")
plt.xlabel("Test case weight parameter")
plt.ylabel("Test case weight")

x = -150
y = 1 + 0.6 * (x / np.abs(x)) / (1 + np.exp(-np.abs(x)/15 + 4.5));
print(y)