import pandas as pd
import numpy as np
np.random.seed = 2021
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

iris = load_iris()
#print ('data contains:',iris.keys())
X, y, labels, feature_names = iris.data, iris.target, iris.target_names, iris['feature_names']
df_iris= pd.DataFrame(X, columns=feature_names)
df_iris['label'] = y
features_dict = {k: v for k, v in enumerate(labels)}
df_iris['label_names'] = df_iris.label.apply(lambda x: features_dict[x])
#print(df_iris)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#print ('X_train.shape=', X_train.shape)
#print ('y_train.shape=', y_train.shape)
#print ('X_test.shape=', X_test.shape)
#print ('y_test.shape=', y_test.shape)
#print ('X_train[0]=')
#print(X_train.iloc[0])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

best_score = 0
n = 0
for i in range(1, X_train.shape[0]+1):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_scaled, y_train)
    if knn.score(X_test_scaled, y_test) > best_score:
        best_score = knn.score(X_test_scaled, y_test)
        n = i

print ('The best k = {} , score = {}'.format(n, best_score))
#flover_dict = dict(zip(df_iris['label'].unique(), df_iris['label_names'].unique()))
#print(flover_dict)
#flover_prediction = knn.predict(scaler.transform([[6.2, 3.4, 5.4, 2.3]]))
#print(flover_dict[flover_prediction[0]])
#print(X_train.shape[0])