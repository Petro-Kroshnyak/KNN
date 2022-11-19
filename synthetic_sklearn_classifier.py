import numpy as np
import plot as draw
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
cmap_bold = ListedColormap(['blue','#FFFF00','black','green'])

np.random.seed = 2021
X_D2, y_D2 = make_blobs(n_samples = 300, n_features = 2, centers = 8,
                       cluster_std = 1.3, random_state = 4)
y_D2 = y_D2 % 2

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)

#print ('X_train.shape= ',X_train.shape)
#print ('y_train.shape= ',y_train.shape)
#print ('X_test.shape= ',X_test.shape)
#print ('y_test.shape= ',y_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
best_score = 0
n = 0

for i in range(1, 226):
    knn = KNeighborsClassifier(n_neighbors=i).fit(X_train_scaled, y_train)
    #print(i, knn_reg.score(X_test_scaled, y_test))
    if knn.score(X_test_scaled, y_test) > best_score:
        best_score = knn.score(X_test_scaled, y_test)
        n = i

print ('The best k = {} , score = {}'.format(n, best_score))

knn = KNeighborsClassifier(n_neighbors=n).fit(X_train_scaled, y_train)
score = knn.score(X_test_scaled, y_test)

draw.boundary_plot(knn, X_train_scaled, y_train, X_test=X_test_scaled, y_test=y_test,
                   title='KNN classification K= {}, score = {:.2f}'.format(n, score))


#plt.figure()
#plt.title('Sample binary classification problem with non-linearly separable classes')
#plt.scatter(X_D2[:,0], X_D2[:,1], c=y_D2,
#           marker= 'o', s=30, cmap=cmap_bold)
#plt.show()

