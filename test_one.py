# %%
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# %%
X, y = make_regression(n_samples=1000, n_features=1, noise=10, shuffle=False, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
x_train_scaler = StandardScaler()
y_test_scaler = StandardScaler()
X_train_scaled = x_train_scaler.fit_transform(X_train)
X_test_scaled = x_train_scaler.transform(X_test)

# %%
model = SGDRegressor(learning_rate='constant', eta0=0.001)
las_model = Lasso(alpha=0.1)
ridge_model = Ridge(alpha=0.1)
elas_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
mse_history = []
max_iter = 1000
for _ in range(max_iter):
    model.partial_fit(X_train_scaled, y_train)
    y_pred = model.predict(X_train_scaled)
    mse = mean_squared_error(y_train, y_pred)
    mse_history.append(mse)

print(f"coef={model.coef_},intercept={model.intercept_}")
# %%
fig = plt.figure()
ax = fig.add_subplot(111)
start, end = 7, 200
ax.plot(range(start, end), mse_history[start:end])
plt.show()



# %%
def check_decreasing(arr):
    for i in range(1, len(arr)):
        if arr[i] >= arr[i - 1]:
            return i  # Return the position where the array stops increasing
    return True  # If the array is strictly increasing

result = check_decreasing(mse_history)
print(result)

# %%
print(mse_history[10:15])
# %%
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X_train, y_train)
ax.plot(X_train, model.predict(X_train_scaled), color='red')
plt.show()

# %%
