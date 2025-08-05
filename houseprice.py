import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

np.random.seed(42)
n_samples = 1000

square_footage = np.random.normal(2000, 500, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)

base_price = 50000
price_per_sqft = 150
price_per_bedroom = 10000
price_per_bathroom = 15000
noise = np.random.normal(0, 25000, n_samples)

prices = (base_price + 
          price_per_sqft * square_footage + 
          price_per_bedroom * bedrooms + 
          price_per_bathroom * bathrooms + 
          noise)

data = pd.DataFrame({
    'SquareFootage': square_footage,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'Price': prices
})

print(data.head())
print(data.describe())

plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

sns.pairplot(data)
plt.show()

print(data.isnull().sum())

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[['SquareFootage', 'Bedrooms', 'Bathrooms']])
X = pd.DataFrame(scaled_features, columns=['SquareFootage', 'Bedrooms', 'Bathrooms'])
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.show()

X_sm = sm.add_constant(X)
sm_model = sm.OLS(y, X_sm).fit()
print(sm_model.summary())

coefficients = pd.DataFrame({
    'Feature': ['Intercept', 'SquareFootage', 'Bedrooms', 'Bathrooms'],
    'Coefficient': [model.intercept_] + list(model.coef_)
})
print(coefficients)

def predict_price(sqft, bedrooms, bathrooms):
    scaled_input = scaler.transform([[sqft, bedrooms, bathrooms]])
    prediction = model.predict(scaled_input)
    return prediction[0]

new_house = [2200, 3, 2]
predicted_price = predict_price(*new_house)
print(f"Predicted price for {new_house[0]} sqft, {new_house[1]} bed, {new_house[2]} bath house: ${predicted_price:,.2f}")

data['Bed_Bath_Ratio'] = data['Bedrooms'] / (data['Bathrooms'] + 1)
data['Price_per_SqFt'] = data['Price'] / data['SquareFootage']
data['SquareFootage_squared'] = data['SquareFootage'] ** 2

from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_score = ridge.score(X_test, y_test)
print(f"Ridge R-squared: {ridge_score:.2f}")

lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
lasso_score = lasso.score(X_test, y_test)
print(f"Lasso R-squared: {lasso_score:.2f}")

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.2f}")