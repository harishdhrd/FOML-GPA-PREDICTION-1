import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("student-data.csv")
data = data.drop(["Empty"], axis=1)

# Split data into features and target
x = data.drop(["GPA"], axis=1)
y = data["GPA"]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=4)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
linear_pred = linear_model.predict(x_test)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

poly_model = LinearRegression()
poly_model.fit(x_train_poly, y_train)
poly_pred = poly_model.predict(x_test_poly)

# Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=4)
rf_model.fit(x_train, y_train)
rf_pred = rf_model.predict(x_test)

# Voting Regressor
voting_model = VotingRegressor([('lr', linear_model), ('poly', poly_model), ('rf', rf_model)])
voting_model.fit(np.column_stack([x_train, x_train_poly]), y_train)
voting_pred = voting_model.predict(np.column_stack([x_test, x_test_poly]))

# Print R2 scores
print(f"Linear Regression R2 Score: {r2_score(y_test, linear_pred)}")
print(f"Polynomial Regression R2 Score: {r2_score(y_test, poly_pred)}")
print(f"Random Forest Regression R2 Score: {r2_score(y_test, rf_pred)}")
print(f"Voting Regressor R2 Score: {r2_score(y_test, voting_pred)}")

# Plotting results
plt.figure(figsize=(14, 7))

# Plot Linear Regression results
plt.subplot(1, 4, 1)
plt.scatter(y_test, linear_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression')

# Plot Polynomial Regression results
plt.subplot(1, 4, 2)
plt.scatter(y_test, poly_pred, color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Polynomial Regression')

# Plot Random Forest Regression results
plt.subplot(1, 4, 3)
plt.scatter(y_test, rf_pred, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Random Forest Regression')

# Plot Voting Regressor results
plt.subplot(1, 4, 4)
plt.scatter(y_test, voting_pred, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Voting Regressor')

# Display the plot
plt.tight_layout()
plt.show()

# Combined plot for comparison
plt.figure(figsize=(7, 7))
plt.scatter(y_test, linear_pred, color='blue', label='Linear Regression')
plt.scatter(y_test, poly_pred, color='red', label='Polynomial Regression')
plt.scatter(y_test, rf_pred, color='green', label='Random Forest Regression')
plt.scatter(y_test, voting_pred, color='purple', label='Voting Regressor')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Model Comparison')
plt.legend()
plt.show()

# Write predicted and actual data to Excel
results = pd.DataFrame({
    'Actual': y_test,
    'Linear_Predicted': linear_pred,
    'Poly_Predicted': poly_pred,
    'RF_Predicted': rf_pred,
    'Voting_Predicted': voting_pred
})

results.to_excel('predicted_vs_actual.xlsx', index=False)
