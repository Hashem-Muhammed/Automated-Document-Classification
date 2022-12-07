from sklearn.cluster import KMeans
from sklearn.ensemble import BaggingClassifier

import DatasetManager as DM
from sklearn.metrics import accuracy_score
model = KMeans(n_clusters=5, init='k-means++', max_iter=1000, n_init=1)
def KMeanss (X):

    model.fit(X)

    print("Clusters:\n")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    vectorizer=DM.vectorizer
    terms = vectorizer.get_feature_names()

    for i in range(5):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind]),
        print

    print("\n")
    result = model.labels_
    print(result)
