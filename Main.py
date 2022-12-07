import KNN as KNN
import DatasetManager as DM
import DocumentPreprocessing as DC
from sklearn.model_selection import train_test_split

datapath = "dataset/BBC News Train.csv"
testpath = "test documents/test4.txt"
X, Y = DM.preprocess(datapath)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=.2)

# KNN Model

KNN.KNN (X_train, X_test, Y_train, Y_test)


print("--------------Document Prediction ----------------")
document=DC.PredictDocument(testpath)

print (KNN.model.predict(document))



