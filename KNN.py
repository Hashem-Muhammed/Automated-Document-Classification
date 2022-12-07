from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

model = KNeighborsClassifier(n_neighbors=17, p=5, metric='euclidean')


def KNN(X_train, X_test, Y_train, Y_test):
    model.fit(X_train, Y_train)
    print("KNN Train SCORE : ", model.score(X_train, Y_train))
    print("KNN Test SCORE : ", model.score(X_test, Y_test))
    Y_pred = model.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred)
    print("Confusion Matrix: \n", cm)
    print("Classification report: \n",classification_report(Y_test, Y_pred))

