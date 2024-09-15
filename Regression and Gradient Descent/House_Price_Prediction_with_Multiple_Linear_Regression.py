import numpy as np

# House features: [Size (sq ft), Number of rooms, Age (years)]
X = np.array([[2100, 3, 20], 
              [1600, 3, 15], 
              [2400, 4, 30], 
              [1416, 2, 20], 
              [3000, 5, 8]], dtype='float32')

# Prices
y = np.array([400000, 330000, 369000, 232000, 539900], dtype='float32')

# adding 1s to our matrix
ones = np.ones(shape=(len(X), 1))
X = np.append(ones, X, axis=1)

# calculating coefficients
coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

# predicting prices
predicted_y = X @ coefficients

# calculating residuals
residuals = y - predicted_y

# calculating total sum of squares
sst = np.sum((y - np.mean(y)) ** 2)

# calculating residual sum of squares
ssr = np.sum(residuals ** 2)

# calculating R^2
r2 = 1 - (ssr/sst)

print("Coefficients:", coefficients)
print("Predicted prices:", predicted_y)
print("R^2:", r2)

# TODO: Add a column of 1's to the matrix X to account for the intercept

# TODO: Calculate the coefficients (beta) using the Normal Equation

# TODO: Print the calculated beta coefficients
