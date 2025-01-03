# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# %%
data_path = './archive/Housing.csv'
df = pd.read_csv(data_path)
df

# %%
house_price = df[['area', 'bedrooms', 'price']]
house_price

# %%
# Define feature (X) and target (y)
X = house_price[['area', 'bedrooms']]
y = house_price[['price']]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train).ravel()

X_test = x_scaler.transform(X_test)
y_test = y_scaler.transform(y_test).ravel()

# %%
print(X_train.shape)
print(y_train.shape)

# %%
model = LinearRegression()
model.fit(X_train, y_train)

coefficients = model.coef_  # Coefficients of the features
intercept = model.intercept_  # Intercept term

print("Coefficients:", coefficients)
print("Intercept:", intercept)


# %%
y_pred = model.predict(X_test)

# %%
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# %%

