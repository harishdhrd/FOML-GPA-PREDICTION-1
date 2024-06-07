import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load and preprocess the data
data = pd.read_csv("student-data.csv")
data = data.drop(["Empty"], axis=1)

x = data.drop(["GPA"], axis=1)
y = data[["GPA"]]

# Print specific and sample data
print(x.loc[x['Reg no'] == 200701107, :].values)
print(x.head())
print(y.head())

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.3, random_state=4)

# Train the linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict the GPA for the test set
pred = model.predict(x_test)

# Calculate and print the R^2 score
print("R^2 score:", r2_score(y_test, pred))

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Perfect Fit Line')
plt.xlabel('Actual GPA')
plt.ylabel('Predicted GPA')
plt.title('Linear Regression: Predicted vs Actual GPA')
plt.legend()
plt.show()
