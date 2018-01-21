from pandas import read_csv
from sklearn.cross_validation import KFold,cross_val_score
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import mean_squared_error


###Load data####

filename = 'pima-indians-diabetes.csv'
names=['preg','plas','pres','skin', 'test', 'mass','pedi','age','class']

data=read_csv(filename,names=names)
array = data.values
# print(array[:2])

X = array[:,:8]
Y = array[:,8]

print(X[0],Y[0])


#create FeatureUnion ##

features = []
features.append(('pca',PCA(n_components=3)))
features.append(('SelectKBest',SelectKBest(k=6)))
features_union = FeatureUnion(features)
print(features_union)
