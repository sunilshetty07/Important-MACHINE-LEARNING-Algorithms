import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
iris_dataset = pd.read_csv('iris1.csv') 
X = iris_dataset[ ['Sepal-Length', 'Sepal-Width', 'Petal-Length', 'Petal-Width'] ]
Y = iris_dataset['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
print("\n X TRAIN \n", X_train)
print("\n X TEST \n", X_test)
print("\n Y TRAIN \n", Y_train)
print("\n Y TEST \n", Y_test)
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train, Y_train)  
accuracy = knn.score(X_test, Y_test)
print("\n TEST SCORE[ACCURACY]: ",accuracy)

for i in range(len(X_test)):
    x = X_test.iloc[i]
    x_new = np.array([x])
    prediction = knn.predict(x_new)
    print("\n Prediction of Test Example {0} ~ Actual : {1} and Predicted :{2}".format( (i+1), Y_test.iloc[i], prediction) )