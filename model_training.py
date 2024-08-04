import pandas as pd
from sklearn.linear_model import ElasticNet
import pickle
from car_data_prep import prepare_data

data = pd.read_csv('dataset.csv')

# Preparing the data for training the model and saving the encoder and scaler
prepared_data = prepare_data(data, fit=True)

# Division into training data and test data
X = prepared_data.drop('Price', axis=1)
y = prepared_data['Price']

# Checking for NaN values and applying them to appropriate values
X.fillna(0, inplace=True)
y.fillna(0, inplace=True)

# Training the model
model = ElasticNet(alpha=0.1, l1_ratio=0.9)
model.fit(X, y)

# Saving the trained model as a PKL file
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(model, file)
