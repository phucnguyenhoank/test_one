# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
data_path = './archive/Housing.csv'
df = pd.read_csv(data_path)
df

# %%
house_price = df[['area', 'price']]
house_price

# %%
def scatter_house_price(house_price, xlabel = 'Area (sq. ft)', ylabel = 'Price (USD)', title = 'Scatter Plot of Area vs. Price'):
    # Create a figure and axis using Matplotlib OOP style
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    ax.scatter(house_price['area'], house_price['price'], color='blue', label='Data Points')

    # Adding labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    # Optional: Adding a grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Adding a legend
    ax.legend()

    # Display the plot
    plt.show()

# %%
scatter_house_price(house_price)

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# %%
# Define feature (X) and target (y)
X = house_price[['area']]  # Feature: 'area'
y = house_price[['price']]   # Target: 'price'

# %%
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
# Standardize the features
scaler_X = StandardScaler()
X_train_standardized = scaler_X.fit_transform(X_train)
X_test_standardized = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_standardized = scaler_y.fit_transform(y_train)
y_test_standardized = scaler_y.transform(y_test)


# %%
# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train_standardized, y_train_standardized)

# %%
# Make predictions on the test data
y_pred_standardized = model.predict(X_test_standardized)

# %%
# Evaluate the model
mse = mean_squared_error(y_test_standardized, y_pred_standardized)
print(f"Mean Squared Error: {mse}")

# %%
# Plotting the results
y_pred = scaler_y.inverse_transform(y_pred_standardized)
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', label='Predicted Prices')

plt.xlabel('Area (sq. ft)')
plt.ylabel('Price (USD)')
plt.title('House Price Prediction')
plt.legend()
plt.show()


# %%
print(y_pred_standardized.shape)
# %%
