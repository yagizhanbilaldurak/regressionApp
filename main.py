from matplotlib import pyplot as plt
from regressor import Regressor
from reader import Reader
from error import Error

predictors = [
    "predictor_1"
]

targets = [
    "target_1" ,
]

X , y = Reader.partial_reading('FilePath', predictors, targets)
model = Regressor()
beta = model.gradient_descent(X, y,learning_rate = 0.000006) #training the model
print(beta)

y_pred = model.predict(X, beta)

mse = Error.mean_squared_error(y, y_pred)
print("Mean Squared Error (MSE):", mse)
r2 = Error.r2_score(y, y_pred)
print("R2 score:", r2)

# Real vs predicted table
plt.scatter(y, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel(", ".join(predictors))
plt.ylabel(", ".join(targets))
plt.title('Model performance')

# MSE info on the plot
plt.text(
    0.05, 0.95,
    f"MSE = {mse:.2f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
)

plt.text(
    0.05, 0.85,
    f"R2 = {r2:.2f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
)
plt.show()