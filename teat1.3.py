import numpy as np
import pandas as pd
df = pd.read_excel("ParsDataSet7.xlsx")
data = df.to_numpy()
X = data[:,[1,2,3,4,5,6,7,8,9]]
y = data[:,-1]
y = np.where((y == 0), 0, 1)
i_class0 = np.where(y == 0)[0]
i_class1 = np.where(y == 1)[0]
n_class0 = len(i_class0)
n_class1 = len(i_class1)
i_class1_upsampled = np.random.choice(i_class1, size=n_class0, replace=True)
np.concatenate((y[i_class1_upsampled], y[i_class0]))
