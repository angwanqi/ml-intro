#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# Load data from sklearn datasets
cancer = datasets.load_breast_cancer()

# Construct features and label
x = cancer.data
y = cancer.target

# Split data into training and test set
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

# Use svm with linear kernel and fit model
clf = svm.SVC(kernel="linear")
clf.fit(x_train, y_train)

# Use model to predict label for test set
y_pred = clf.predict(x_test)

# Calculate accuracy of predictions
acc = metrics.accuracy_score(y_test, y_pred)

print(acc)


# In[ ]:




