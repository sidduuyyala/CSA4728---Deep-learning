import numpy as np
import matplotlib.pyplot as plt

# Mean Squared Error (Cost function)
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Gradient Descent function
def gradient_descent(x, y, lr=0.0001, iterations=2000):
    w, b = 0.1, 0.01  # Initialize weight and bias
    n = len(x)        # Number of data points
    
    for i in range(iterations):
        y_pred = w * x + b  # Predict the output
        cost = mse(y, y_pred)  # Calculate the cost (MSE)
        
        # Calculate gradients
        w_gradient = -(2/n) * np.dot(x, (y - y_pred))
        b_gradient = -(2/n) * np.sum(y - y_pred)
        
        # Update the weights and bias
        w -= lr * w_gradient
        b -= lr * b_gradient
        
        if i % 100 == 0:  # Print every 100th iteration
            print(f"Iteration {i}: Cost = {cost}, Weight = {w}, Bias = {b}")
    
    return w, b

# Main function
def main():
    # Modified Data (e.g., adding noise to simulate a different scenario)
    X = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0])
    Y = np.array([12.0, 22.0, 32.0, 38.0, 52.0, 54.0, 63.0, 72.0, 78.0, 92.0])  # Slightly different from actual data
    
    # Estimate weight and bias using gradient descent
    w, b = gradient_descent(X, Y, lr=0.0001, iterations=2000)
    print(f"\nEstimated Weight: {w}, Estimated Bias: {b}")
    
    # Predictions and Plotting
    Y_pred = w * X + b
    
    # Plot modified data and regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, color='green', label='Modified Data')
    plt.plot(X, Y_pred, color='orange', label='Fitted Line')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Gradient Descent with Modified Data")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
