import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import matplotlib.patches as mpatches


def boundary_plot(knn, X_train, y_train, X_test=None, y_test=None, title=None, precision=0.01, size=50):
    points_colors = mpl.cm.viridis(.90), mpl.cm.viridis(.26)
    areas_colors = mpl.cm.viridis(.95), mpl.cm.viridis(.23)
    cmap_light = ListedColormap(areas_colors)
    cmap_bold = ListedColormap(points_colors)

    step = precision

    X = np.concatenate([X_train, X_test], axis=0)
    x1_min, x1_max = X[:, 0].min() - .1, X[:, 0].max() + 0.1
    x2_min, x2_max = X[:, 1].min() - .1, X[:, 1].max() + 0.1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, step),
                           np.arange(x2_min, x2_max, step))

    target_grid = (np.c_[xx1.ravel(), xx2.ravel()])

    Z = knn.predict(target_grid)

    Z = Z.reshape(xx1.shape)
    plt.figure()
    plt.pcolormesh(xx1, xx2, Z, cmap=cmap_light)

    plt.scatter(X_train[:, 0], X_train[:, 1], s=size, c=y_train.ravel(), cmap=cmap_bold, edgecolor='black')
    plt.scatter(X_test[:, 0], X_test[:, 1], marker='^', s=size, c=y_test.ravel(), cmap=cmap_bold,
                edgecolor='black')
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    patch0 = mpatches.Patch(color=points_colors[0], label='negative')
    patch1 = mpatches.Patch(color=points_colors[1], label='positive')
    plt.legend(handles=[patch0, patch1])
    plt.title(title)

    plt.xlabel('feature 1')
    plt.ylabel('feature 2')

    plt.show()
