from sklearn import linear_model
import numpy as np

x = np.array([[1], [2], [3], [4], [5]])
y = np.array([15000, 20000, 25000, 30000, 35000])

model = linear_model.LinearRegression()
model.fit(x, y)

x_new = np.array([6]).reshape((-1, 1))
y_new = model.predict(x_new)
print(y_new)
