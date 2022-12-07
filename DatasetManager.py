import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
def preprocess( path):
    dataset = pd.read_csv(path)
    dataset['category_id'] = dataset['Category'].factorize()[0]
    X = dataset.iloc[:, 1].values
    Y = dataset.iloc[:, 2].values

    X = vectorizer.fit_transform(dataset.Text).toarray()
    labelencoder_Y = LabelEncoder()
    Y = labelencoder_Y.fit_transform(Y)
    le_name_mapping = dict(zip(labelencoder_Y.classes_, labelencoder_Y.transform(labelencoder_Y.classes_)))
    print(le_name_mapping)
    return X, Y