import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("RegData.csv")
x = df["x"].values.reshape(-1, 1)
y = df["y"].values.reshape(-1, 1)
m = len(x)
plt.scatter(x, y)
plt.show()
theta0 = 0
theta1 = 0
alpha = 0.001
cost = []
for _ in range(1000):
    hypothesis = np.array([(theta0 + theta1 * i) for i in x])
    loss_function = (1/2*m) * sum((hypothesis - y)**2)
    cost.append(loss_function)
    delta_theta0 = (1/m) * sum(hypothesis - y)
    delta_theta1 = (1/m) * sum((hypothesis - y) * x)
    theta0 = theta0 - alpha * delta_theta0
    theta1 = theta1 - alpha * delta_theta1
print(theta0, theta1)
plt.plot(np.arange(1000), cost)
plt.show()
new_hypothesis = [theta0 + theta1 * i for i in x]
plt.plot(x, new_hypothesis, 'r')
plt.scatter(x, y)
plt.show()