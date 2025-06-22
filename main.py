from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

data = fetch_california_housing()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="MedHouseVal")

model = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=42)
model.fit(X_train, y_train)

# Evaluate accuracy of model:

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Sample prediction:

# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
sample = [[8.3252, 41.0, 6.9841, 1.0238, 322.0, 2.5556, 37.88, -122.23]]
sample_df = pd.DataFrame(sample, columns=X.columns)

prediction = model.predict(sample_df)
print("Predicted price: $", round(prediction[0] * 100000, 2))
