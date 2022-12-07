from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
DecisionTreeClassifierModel = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=100, min_samples_leaf=5)
def DecisionTree(X_train,X_test, Y_train,Y_test):

    rf = RandomForestClassifier(n_estimators=10)
    DecisionTreeClassifierModel.fit(X_train, Y_train)
    rf.fit(X_train, Y_train)
    print('DecisionTreeClassifierModel Train Score is : ' , DecisionTreeClassifierModel.score(X_train, Y_train))
    print('DecisionTreeClassifierModel Test Score is : ', DecisionTreeClassifierModel.score(X_test, Y_test))
    print('DecisionTreeClassifierModel Classes are : ', DecisionTreeClassifierModel.classes_)
    print('DecisionTreeClassifierModel feature importances are : ', DecisionTreeClassifierModel.feature_importances_)
    y_pred = DecisionTreeClassifierModel.predict(X_test)
    y_pred
    print(classification_report(Y_test, y_pred))
    CM = confusion_matrix(Y_test, y_pred)
    print('Confusion Matrix is : \n', CM)
    print("After Applying RandomForest ensemble learning")
    print("RandomForest score : ",rf.score(X_test,Y_test))

