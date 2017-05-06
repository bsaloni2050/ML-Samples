import pandas as pd
from sklearn.cluster import KMeans
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from pylab import *


# Load dataset
path = "/Users/salonibindra/Downloads/green_tripdata_2016-02.csv"
columns = ['Pickup_longitude' ,	'Pickup_latitude',	'Dropoff_longitude',	'Dropoff_latitude',	'Passenger_count',	'Trip_distance'	,'Fare_amount',	'Extra',	'Tip_amount',	'Tolls_amount',	'Total_amount',	'Payment_type',	'Trip_type ']

dataset = pd.read_csv(path, names=columns, low_memory=False)

print("Shape of Data :")
print(dataset.shape)
dataset = dataset.fillna(lambda x: x.median())
for i in columns:
    dataset[i] = pd.to_numeric(dataset[i], errors='coerce')


kmeans = KMeans(n_clusters=100, random_state=0).fit(dataset)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_


print("List of Centroids :")

# print(labels)
print(centroids)

for i in centroids:
    print(i)

#print(dataset.head(20))

# # Split-out validation dataset
# array = dataset.values
# X = array[:,0:4]
# Y = array[:,4]
# validation_size = 0.20
# seed = 7
# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
#
# # Test options and evaluation metric
# seed = 7
# scoring = 'accuracy'
#
# # Spot Check Algorithms
# models = []
# models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC()))
# # evaluate each model in turn
# results = []
# names = []
# for name, model in models:
#     kfold = model_selection.KFold(n_splits=10, random_state=seed)
#     cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)
