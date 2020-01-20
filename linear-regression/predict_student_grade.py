#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# Read in student data from file
data = pd.read_csv("student-mat.csv", sep=";")

# Select columns to use
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Set the label to predict
predict = "G3"

# Construct Features and Labels Arrays
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# Split data into training (90%) and test set (10%)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

# Use linear regression model
linear = linear_model.LinearRegression()

# Fit the model using training data
linear.fit(x_train, y_train)

# Get accuracy of model
acc = linear.score(x_test, y_test)
print("Accuracy:",acc)


# In[6]:


predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


# In[ ]:




