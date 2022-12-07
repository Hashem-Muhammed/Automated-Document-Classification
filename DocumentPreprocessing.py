from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
stop_words = set(stopwords.words('english'))
stemmer = WordNetLemmatizer()
import DatasetManager as DS

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
        if (document not in stop_words):
            # print(document)
            documents.append(document)


def listToString(s):
    str1 = ""

    for ele in s:
        str1 = str1 + ele + " "

    return str1
def PredictDocument(path):
    vectorizer = DS.vectorizer
    f = open(path, "rt")
    test = f.read()
    f.close()
    wordss = []
    words = test.split()
    preprocessing(wordss, words)
    TestStr = listToString(wordss)
    TestStr = vectorizer.transform([TestStr]).toarray()

    return TestStr