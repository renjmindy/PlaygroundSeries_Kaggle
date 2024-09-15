import numpy as np
import math

# Sample house sizes in square feet, standardized
house_sizes = np.array([[1000], [1500], [2000]])
house_sizes = (house_sizes - np.mean(house_sizes)) / np.std(house_sizes)
# Sample house prices in 1000s of dollars
house_prices = np.array([[300], [450], [600]])
# We initialize our parameters: slope (a) and intercept (b)
theta_real_estate = np.random.rand(2, 1)
# Learning rate and iterations for gradient descent, adjusted learning rate
alpha_real_estate = 0.01
iterations = 500
# Add a column of ones to the house sizes to accommodate the intercept (b)
X_b_real_estate = np.c_[np.ones((len(house_sizes), 1)), house_sizes]

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/2*m) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    for i in range(iterations):  # Iterate until convergence
        prediction = np.dot(X, theta)  # Matrix multiplication between X and theta
        # Gradient update rule with correct cost function calculation
        theta = theta - (1/m) * alpha * (X.T.dot(prediction - y))
        cost_history[i] = compute_cost(X, y, theta)
    return theta, cost_history

def cost_threshold(X, y, theta, alpha, threshold):
    m = len(y)
    cost_history = list()
    cost_delta = math.inf
    while cost_delta > threshold:
        prediction = np.dot(X, theta)
        theta -= (1/m) * alpha * (X.T.dot(prediction - y))
        cost_history.append(compute_cost(X, y, theta))
        
        if len(cost_history) > 1:
            cost_delta = abs(cost_history[-2] - cost_history[-1])
            
    return theta, cost_history

# Run gradient descent
#theta_real_estate, cost_history = gradient_descent(X_b_real_estate, house_prices, theta_real_estate, alpha_real_estate, iterations)
theta_real_estate, cost_history = cost_threshold(X_b_real_estate, house_prices, theta_real_estate, alpha_real_estate, iterations)

for i, cost in enumerate(cost_history[::10]):
    print(f'Iteration {i * 10}: Cost = {cost}')
