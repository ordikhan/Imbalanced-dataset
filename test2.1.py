from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
from numpy import where
# define dataset
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_excel("ParsDataSet7.xlsx")

# print(df.head())

data = df.to_numpy()

X = data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]]
y = data[:, -1]

# summarize class distribution
counter = Counter(y)
print(counter)
# # define pipeline
over = SMOTE(sampling_strategy=0.22)
under = RandomUnderSampler(sampling_strategy=0.1)
model = DecisionTreeClassifier()

steps = [('over', over), ('under', under), ('model', model)]
pipeline = Pipeline(steps=steps)
# # transform the dataset
X, y = pipeline.fit_resample(X, y)
# # summarize the new class distribution
# counter = Counter(y)
# print(counter)
# # scatter plot of examples by class label
# for label, _ in counter.items():
# 	row_ix = where(y == label)[0]
# 	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
# pyplot.legend()
# pyplot.show()
