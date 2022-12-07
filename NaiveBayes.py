
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
GaussianNBModel = GaussianNB()
def NaiveBayes(X_train,X_test, Y_train,Y_test):

    GaussianNBModel.fit(X_train, Y_train)
    print("Gaussian train model is", GaussianNBModel.score(X_train, Y_train))
    print("Gaussian test ٍmodel is", GaussianNBModel.score(X_test, Y_test))
    print("----------------------------------------------------------------")
    # calculating predictions
    y_pred = GaussianNBModel.predict(X_test)
    from sklearn.metrics import classification_report
    print(classification_report(Y_test, y_pred))

    BG= BaggingClassifier(GaussianNBModel,max_samples=.2    ,max_features=1.0,n_estimators=20,random_state=8)
    BG.fit(X_train, Y_train)
    print("After applying BaggingClassifier ensemble learning")
    print("Bagging Score: ", BG.score(X_test, Y_test))

