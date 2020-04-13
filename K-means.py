import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
import sklearn.metrics as sm
import pandas as pd
import numpy as np
#%matplotlib inline
# Load the iris dataset
iris_dataset = pd.read_csv('Iris.csv')
iris_dataset['Targets'] = iris_dataset.Class.map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
X = iris_dataset[['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width' ] ]
Y = iris_dataset[['Targets']]
#Build K-Means Model
model = KMeans(n_clusters = 3, random_state = 0)
model.fit(X)
print('Model Labels:\n', model.labels_ )
#Build GMM for EM algorithm
scaler = preprocessing.StandardScaler()
scaler.fit(X)
xs = scaler.transform(X)
gmm = GaussianMixture(n_components = 3, random_state = 0)
gmm.fit(xs)
Y_gmm = gmm.predict(xs)
print('GMM Labels:\n', Y_gmm)
#Visualize the clustering results
# To View the results set the size of the plot
plt.figure(figsize = (14, 14))
# Create a colormap
colormap = np.array(['red', 'lime', 'black'])
# Plot Orginal using Petal features
plt.subplot(2, 2, 1)
plt.scatter(X.Petal_Length,X.Petal_Width, c = colormap[Y.Targets], s=40)
plt.title('Real Classification')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
# Plot the K-Means model classifications
plt.subplot(2, 2, 2)
plt.scatter(X.Petal_Length,X.Petal_Width, c = colormap[model.labels_], s=40)
plt.title('K Means Clustering')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
#Plot the GMM Model classification
print("\n")
print("\n")
plt.subplot(2, 2, 3)
plt.scatter(X.Petal_Length, X.Petal_Width, c = colormap[Y_gmm], s=40)
plt.title('GMM Based Clustering')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
#Calculate performance metrics for K-Means and GMM
print('Evaluation of K-Means with ground truth classification of Iris Dataset')
print('Rand Index:%f ' % sm.adjusted_rand_score(Y.Targets, model.labels_ ))
print('Homogenity Score:%f ' % sm.homogeneity_score(Y.Targets, model.labels_ ))
print('Completeness Score:%f ' % sm.completeness_score(Y.Targets, model.labels_ ))
print('V-Measure:%f ' % sm.v_measure_score(Y.Targets, model.labels_ ))
print('Evaluation of GMM with ground truth classification of Iris Dataset')
print('Rand Index:%f ' % sm.adjusted_rand_score(Y.Targets, Y_gmm))
print('Homogenity Score:%f ' % sm.homogeneity_score(Y.Targets, Y_gmm))
print('Completeness Score:%f ' % sm.completeness_score(Y.Targets, Y_gmm))
print('V-Measure:%f ' % sm.v_measure_score(Y.Targets, Y_gmm))