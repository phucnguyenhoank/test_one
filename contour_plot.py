# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error

# %%
data_path = './archive/Housing.csv'
df = pd.read_csv(data_path)
df

# %%
house_price = df[['area', 'bedrooms', 'price']]
house_price


# %%
def contour_house_price(house_price, xlabel = 'Area (sq. ft)', ylabel = 'Bedroom', zlabel = 'Price (USD)', title = 'Contour Plot of Area, Bedroom vs. Price'):
    X = house_price[['area', 'bedrooms']].values
    y = house_price['price'].values
    bo = 9999
    # Generate a grid of w1 and w2 values
    w1 = np.linspace(-bo, bo, 100)  # Range for w1 (parameter of area)
    w2 = np.linspace(-bo, bo, 100)  # Range for w2 (parameter of bedrooms)
    w1_grid, w2_grid = np.meshgrid(w1, w2)

    # Initialize J(w1, w2) grid
    J_grid = np.zeros_like(w1_grid)

    # Compute J(w1, w2) for each combination of w1 and w2
    for i in range(w1_grid.shape[0]):
        for j in range(w1_grid.shape[1]):
            # Calculate predicted prices
            predicted_prices = w1_grid[i, j] * X[:, 0] + w2_grid[i, j] * X[:, 1]
            
            # Compute MSE
            J_grid[i, j] = mean_squared_error(y, predicted_prices)

    # Plot the contour
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contour(w1_grid, w2_grid, J_grid, levels=10, cmap='viridis')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Mean Squared Error (J)')

    # Add labels and title
    ax.set_xlabel('w1 (parameter for area)')
    ax.set_ylabel('w2 (parameter for bedrooms)')
    ax.set_title('Contour Plot of J(w1, w2)')

    plt.show()


# %%
contour_house_price(house_price)

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

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
y_train_standardized = scaler_y.fit_transform(y_train).ravel()
y_test_standardized = scaler_y.transform(y_test).ravel()

# %%
# Create and train the model
model = SGDRegressor(loss='squared_error', max_iter=6000)
model.fit(X_train_standardized, y_train_standardized)


# %%
# Evaluate the model
# Make predictions on the test data
y_pred_standardized = model.predict(X_test_standardized)
mse = mean_squared_error(y_test_standardized, y_pred_standardized)

print(f"Final Mean Squared Error (on test data): {mse}")


# %%
# Plot the test data and the fitted line
y_pred = scaler_y.inverse_transform(y_pred_standardized.reshape(-1, 1))
plt.scatter(X_test, y_test, color='blue', label='Test data')
plt.plot(X_test, y_pred, color='red', label='Fit line')

# Add labels and title
plt.xlabel('Area (feet^2)')
plt.ylabel('Price (USD)')
plt.title('Test Data and Fit Line')
plt.legend()

# Show the plot
plt.show()


# %%
