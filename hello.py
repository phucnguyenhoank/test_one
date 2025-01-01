# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
data_path = './archive/Housing.csv'
df = pd.read_csv(data_path)
df

# %%
house_price = df[['area', 'bedrooms', 'price']]
house_price

# %%
def visualize_house_price(data):
    """
    Visualize the relationship between area, bedrooms, and price in a 3D scatter plot.
    
    Parameters:
        data (DataFrame): A DataFrame containing 'area', 'bedrooms', and 'price'.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extracting the data
    area = data['area']
    bedrooms = data['bedrooms']
    price = data['price']
    
    # Creating the scatter plot
    scatter = ax.scatter(area, bedrooms, price, c=price, cmap='viridis', marker='o')
    
    # Adding labels and title
    ax.set_title('House Prices vs. Area and Bedrooms', fontsize=14)
    ax.set_xlabel('Area (sq ft)', fontsize=12)
    ax.set_ylabel('Number of Bedrooms', fontsize=12)
    ax.set_zlabel('Price (USD)', fontsize=12)
    
    # Adding color bar to represent price values
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Price (USD)', fontsize=12)
    
    plt.show()

# %%
visualize_house_price(house_price)

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# %%
# Define feature (X) and target (y)
X = house_price[['area', 'bedrooms']]
y = house_price[['price']]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
model = LinearRegression()
model.fit(X_train, y_train)

coefficients = model.coef_  # Coefficients of the features
intercept = model.intercept_  # Intercept term

print("Coefficients:", coefficients)
print("Intercept:", intercept)

# %%
import numpy as np

# %%
def compute_cost(w1, w2, X, y):
    """
    Compute the Mean Squared Error cost function for given parameters w1 and w2.
    
    Parameters:
        w1 (float): Weight for area.
        w2 (float): Weight for bedrooms.
        X (ndarray): Input features (area and bedrooms).
        y (ndarray): Actual prices.
    
    Returns:
        float: Computed cost function value.
    """
    m = len(y)  # Number of data points
    predictions = w1 * X[:, 0] + w2 * X[:, 1] + 478507.59000274 # Linear model: w1 * area + w2 * bedrooms
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)  # MSE (Mean Squared Error)
    return cost

def visualize_cost_function(X, y):
    """
    Visualize the cost function J(w1, w2) for different values of w1 and w2 as a contour plot.
    
    Parameters:
        X (DataFrame or ndarray): Input features (area and bedrooms).
        y (ndarray): Actual prices.
    """
    w1_opt = 433.8
    w2_opt = 697941.2
    
    # Set a smaller range around the optimized values
    w1_vals = np.linspace(w1_opt - 100, w1_opt + 100, 100)  # Range for w1 around the optimal value
    w2_vals = np.linspace(w2_opt - 100000, w2_opt + 100000, 100) # Range for w2 around the optimal value
    
    # Create a meshgrid for the contour plot
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    J_vals = np.zeros(W1.shape)  # To store cost values
    
    # Compute the cost J(w1, w2) for each pair (w1, w2)
    for i in range(len(w1_vals)):
        for j in range(len(w2_vals)):
            J_vals[i, j] = compute_cost(w1_vals[i], w2_vals[j], X, y)
    
    # Plot the contour plot
    plt.figure(figsize=(10, 7))
    contour = plt.contour(W1, W2, J_vals, 50, cmap='viridis')
    plt.colorbar(contour)
    
    # Add labels and title
    plt.title('Contour Plot of Cost Function J(w1, w2)', fontsize=14)
    plt.xlabel('w1 (Weight for Area)', fontsize=12)
    plt.ylabel('w2 (Weight for Bedrooms)', fontsize=12)
    
    plt.show()

# %%
X = house_price[['area', 'bedrooms']].values
y = house_price['price'].values
visualize_cost_function(X, y)

# %%
def visualize_house_price_with_line(data, w1, w2, intercept):
    """
    Visualize the relationship between area, bedrooms, and price in a 3D scatter plot,
    and plot the regression plane based on the given weights and intercept.
    
    Parameters:
        data (DataFrame): A DataFrame containing 'area', 'bedrooms', and 'price'.
        w1 (float): Coefficient for area.
        w2 (float): Coefficient for bedrooms.
        intercept (float): Intercept term.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extracting the data
    area = data['area']
    bedrooms = data['bedrooms']
    price = data['price']
    
    # Creating the scatter plot
    scatter = ax.scatter(area, bedrooms, price, c=price, cmap='viridis', marker='o')
    
    # Adding labels and title
    ax.set_title('House Prices vs. Area and Bedrooms', fontsize=14)
    ax.set_xlabel('Area (sq ft)', fontsize=12)
    ax.set_ylabel('Number of Bedrooms', fontsize=12)
    ax.set_zlabel('Price (USD)', fontsize=12)
    
    # Adding color bar to represent price values
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Price (USD)', fontsize=12)
    
    # Plotting the regression plane
    # Create a grid of area and bedrooms values for the plane
    area_grid, bedrooms_grid = np.meshgrid(np.linspace(area.min(), area.max(), 30),
                                           np.linspace(bedrooms.min(), bedrooms.max(), 30))
    
    # Compute the predicted prices from the regression model
    predicted_prices = w1 * area_grid + w2 * bedrooms_grid + intercept
    
    # Plot the plane
    ax.plot_surface(area_grid, bedrooms_grid, predicted_prices, alpha=0.5, cmap='coolwarm')
    
    plt.show()
    

# %%
w1 = 417.53813
w2 = 796537.239
intercept = 298016.8074967
visualize_house_price_with_line(house_price, w1, w2, intercept)
# %%
