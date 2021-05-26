import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
from sympy import symbols
# x, y = symbols('x, y')
df = pd.read_csv("data.csv")

# def linear_regression(inner_y, inner_x, inner_m):
#     global theta0, theta1
#     h = [theta1[i] * x + theta0[i] for i in range(m)]
#     total_sum = 0
#     gradient_square = 0
#     for j in range(inner_m):
#         h[j] = h[j].subs(x, inner_x[j])
#         total_sum += (h[j] - inner_y[j]) ** 2
#         gradient_square += (inner_x[j] * ((h[j] - inner_y[j]) ** 2) / inner_m) ** 2
#     gradient = math.sqrt(gradient_square)
#     print(gradient)
#     if gradient == 0:
#         print(theta0, theta1)
#         return
#     else:
#         for j in range(inner_m):
#             theta0[j] = theta0[j] - (alpha * 1 / inner_m) * total_sum
#             theta1[j] = theta1[j] - ((alpha * 1 / inner_m) * total_sum) * inner_x[j]
#
# x_input = df["x"].values
# y_input = df["y"].values
# theta1 = np.ones(len(x_input))
# theta0 = np.ones(len(x_input))
# alpha = 0.001
# m = len(x_input)
# for i in range(1000):
#     linear_regression(inner_y=y_input, inner_x=x_input, inner_m=m)

x = df["x"].values
y = df["y"].values
plt.scatter(x, y)
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
model = LinearRegression()
model.fit(x, y)
slope = model.coef_[0]
intercept = model.intercept_
plt.plot(x, model.predict(x))
r2 = r2_score(x, y)
print(r2)
plt.show()