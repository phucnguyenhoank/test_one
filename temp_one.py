
# # List to store the loss at each iteration
# loss_history = []

# # Manually run the training for each iteration to capture loss
# for epoch in range(3000):
#     model.partial_fit(X_train, y_train)  # Fit on the data for one iteration
    
#     # Get predictions for the current model
#     predictions = model.predict(X_train)
    
#     # Calculate Mean Squared Error (loss) for the current iteration
#     mse_loss = mean_squared_error(y_train, predictions)
    
#     # Append the loss to the history
#     loss_history.append(mse_loss)

# # Plotting the loss curve
# plt.plot(loss_history)
# plt.title('Loss at Each Iteration')
# plt.xlabel('Iteration')
# plt.ylabel('Loss (MSE)')
# plt.show()