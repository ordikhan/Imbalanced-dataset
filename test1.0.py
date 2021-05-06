# Load libraries
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()

# Create feature matrix
X = iris.data

# Create target vector
y = iris.target
# Remove first 40 observations
X = X[40:,:]
y = y[40:]
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
i_class0_upsampled = np.random.choice(i_class0, size=300, replace=True)
# i_class1_upsampled = np.random.choice(i_class1, size=300, replace=True)

print(i_class0_upsampled)
# Join together class 0's upsampled target vector with class 1's target vector
np.concatenate((y[i_class0_upsampled], y[i_class1]))
print(len(np.concatenate((y[i_class0_upsampled], y[i_class1]))))
print(y[i_class1])
print(y[i_class0])
# print("hjk",y[i_class1_upsampled])