import numpy as np
import pandas as pd

# Step 1: Sample Dataset
data = {
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'Married': ['Yes', 'No', 'Yes', 'No', 'Yes'],
    'ApplicantIncome': [5000, 3000, 4000, 2500, 6000],
    'LoanAmount': [200, 100, 150, 120, 250],
    'Credit_History': [1, 0, 1, 0, 1],
    'Loan_Status': [1, 0, 1, 0, 1]  # 1 = Approved, 0 = Not Approved
}
df = pd.DataFrame(data)

# Step 2: Encode Categorical Variables
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})

# Step 3: Prepare Data
X = df[['Gender', 'Married', 'ApplicantIncome', 'LoanAmount', 'Credit_History']].values
y = df['Loan_Status'].values.reshape(-1, 1)

# Normalize features
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Add bias (intercept) term
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Step 4: Logistic Regression Functions

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, weights):
    m = len(y)
    predictions = sigmoid(X @ weights)
    cost = (-1/m) * np.sum(y * np.log(predictions + 1e-9) + (1 - y) * np.log(1 - predictions + 1e-9))
    return cost

def gradient_descent(X, y, weights, lr, epochs):
    m = len(y)
    for i in range(epochs):
        predictions = sigmoid(X @ weights)
        error = predictions - y
        gradient = (1/m) * X.T @ error
        weights -= lr * gradient

        if i % 100 == 0:
            print(f"Epoch {i}, Cost: {compute_cost(X, y, weights):.4f}")
    return weights

# Step 5: Initialize Parameters
weights = np.zeros((X.shape[1], 1))
learning_rate = 0.1
epochs = 1000

# Step 6: Train Model
weights = gradient_descent(X, y, weights, learning_rate, epochs)

# Step 7: Prediction Function
def predict(X, weights):
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # normalize
    X = np.hstack((np.ones((X.shape[0], 1)), X))       # add bias
    probs = sigmoid(X @ weights)
    return (probs >= 0.5).astype(int)

# Step 8: Test Predictions
y_pred = predict(df[['Gender', 'Married', 'ApplicantIncome', 'LoanAmount', 'Credit_History']].values, weights)
print("Predicted Loan Status:", y_pred.ravel())
print("Actual Loan Status:   ", y.ravel())
