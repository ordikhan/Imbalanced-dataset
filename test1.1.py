# Load libraries
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

df = pd.read_excel("ParsDataSet7.xlsx")

print(df.head())

data = df.to_numpy()

X = data[:,[1,2,3,4,5,6,7,8,9]]
y = data[:,-1]


# Create feature matrix

# Create target vector
# Remove first 40 observations

print("yy",len(y))
# Create binary target vector indicating if class 0
y = np.where((y == 0), 0, 1)

# Look at the imbalanced target vector
print(len(y))


# Indicies of each class' observations
i_class0 = np.where(y == 0)[0]
i_class1 = np.where(y == 1)[0]

# Number of observations in each class
n_class0 = len(i_class0)
n_class1 = len(i_class1)
print("n_class0",n_class0)
print("n_class1",n_class1)
# For every observation in class 1, randomly sample from class 0 with replacement
i_class1_upsampled = np.random.choice(i_class1, size=n_class0, replace=True)
print(i_class1_upsampled)
# Join together class 0's upsampled target vector with class 1's target vector
np.concatenate((y[i_class1_upsampled], y[i_class0]))
print(len(np.concatenate((y[i_class1_upsampled], y[i_class0]))))
print(len(y[i_class1_upsampled]))