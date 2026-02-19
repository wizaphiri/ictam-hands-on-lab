# Create a simple linear regression model to predict salary given years of experience

# 1. import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


# 2. get data
df = pd.read_csv('salaries.csv')


# 3. Explore and Understand (1 command after another)
print(df.shape)
print(df.describe())
print(df.info())
print(df.corr()) # High positive correlation expected (~0.95+)

plt.scatter(df.years_of_experience, df.salary)
plt.title("Correlation (years_of_experience & salary)")
plt.show()


# 4. Prepare Data

print(f"Missing values:\n{df.isnull().sum()}") # Check for missing values
df = df.dropna() # Drop missing values if any existed

# Reshape for Scikit-Learn (X must be a 2D array)
X = df[['years_of_experience']] 
y = df['salary']

# # Split into Training and Testing sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 5. Create Model & Train (This is where the computer learns the relationship: salary = m(experience) + b )
model = LinearRegression()
model.fit(X, y) # model.fit(X_train, y_train) -> use full dataset

# Print model parameters
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

# Make predictions on the test set
y_pred = model.predict(X) # X_test


# 6. Evaluate (r2 and MAE)
r2 = r2_score(y, y_pred) # r2_score(y_test, y_pred)
mae = mean_absolute_error(y, y_pred) # mean_absolute_error(y_test, y_pred)

print(f"R-Squared Score: {r2:.4f}")
print(f"Mean Absolute Error: ${mae:.2f}")

# test prediction again with known value
test_experience_df = pd.DataFrame({'years_of_experience': [5]}) # create df for test value to match the training format
predicted_salary = model.predict(test_experience_df) # predict using the DataFrame
print(f"Predicted Salary for {test_experience_df.iloc[0, 0]} years: ${predicted_salary[0]:,.2f}")

