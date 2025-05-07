import numpy as np

def manhattan_distance(X_train, x_test_item):

    return np.sum(np.abs(X_train - x_test_item), axis=1)

def knn_1(X_train, y_train, X_test, k):

    y_train = np.array(y_train)

    predicted = []

    for x_test_item in X_test:
        
        distances = manhattan_distance(X_train, x_test_item)
        k_distances = np.argsort(distances)[:k]
        labels = y_train[k_distances]
        max_label = np.argmax(np.bincount(labels))

        predicted.append(max_label)

    return predicted