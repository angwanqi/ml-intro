#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

# Read in car data
data = pd.read_csv("car.data")

# Preprocess data so that only contains integers
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

# Set the label to predict
predict = "class"

# Construct the features and label 
X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

# Split data into training and test set
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

# Use K-Nearest Neighbours Model
model = KNeighborsClassifier(n_neighbors=9)

# Fit the model with training data
model.fit(x_train, y_train)

# Get accuracy of model
acc = model.score(x_test, y_test)
print(acc)


# In[ ]:




