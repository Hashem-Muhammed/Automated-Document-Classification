import numpy as np
import pandas as pd
import os
import re

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_word = set(stopwords.words('english'))
stemmer = WordNetLemmatizer()


def preprocessing(documents, X):
    for sen in range(0, len(X)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
        # document =  re.sub(r"\b[a-zA-Z]\b", "", document)
        document = re.sub('\[[^]]*\]', '', document)
        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()

        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        if (document not in stop_word):
            # print(document)
            documents.append(document)


def listToString(s):
    str1 = ""

    for ele in s:
        str1 = str1 + ele + " "

    return str1


#os.listdir("D:/S H E R I F/Sherif/6th semester/Selected 2/Project/Automated Classification")
df_train = pd.read_csv(
    "dataset/BBC News Train.csv")

df_train['category_id'] = df_train['Category'].factorize()[0]


df_train['category_id'][0:10]
df_train.head(10)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
features = tfidf.fit_transform(df_train.Text).toarray()
labels = df_train.category_id

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=4)

from sklearn import svm

test_file = open("test documents/test4.txt", "rt")
testing = test_file.read()
test_file.close()
list1 = []
list2 = testing.split()
preprocessing(list2, list1)
tostring_test = listToString(list2)
tostring_test = tfidf.transform([tostring_test]).toarray()
model = svm.SVC(kernel='linear')

model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

file_predict = model.predict(tostring_test)
y_predict = model.predict(X_test)
print(file_predict)
print(accuracy_score(y_test, y_predict))

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_predict))

from sklearn.metrics import classification_report

print(classification_report(y_test, y_predict))

