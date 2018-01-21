
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
filename = 'pima-indians-diabetes.csv'
names=['preg','plas','pres','skin', 'test', 'mass','pedi','age','class']

data=pd.read_csv(filename,names=names)
array = data.values
# print(array[:2])

X = array[:,:8]
Y = array[:,8]

print(X[0],Y[0])


# In[6]:


from sklearn import cross_validation
num_folds = 10
num_instances = len(X)
seed = 7

kfold = cross_validation.KFold(n=num_instances,n_folds=num_folds,random_state = seed)


# In[4]:


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#prepare models
models = []
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))


# In[10]:


results = []
names = []
scoring = "accuracy"

for name,model in models:
    cv_result = cross_validation.cross_val_score(model,X,Y,cv=kfold,scoring=scoring)
    results.append(cv_result)
    names.append(name)
    msg = "%s: %f (%f)"%(name,cv_result.mean(),cv_result.std())
    print(msg)


# In[11]:


fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

