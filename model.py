import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

#Step 1

df = pd.read_csv("static/dataset.csv")
X = df["CONTENT"]
y = df["CLASS"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

clf = LogisticRegression()
clf.fit(X_train_vectorized, y_train)

predictions = clf.predict(X_test_vectorized)
print("Machine learning model trained")
print("Accuracy:", accuracy_score(y_test, predictions))

joblib.dump(clf, "static/models/spam_classifier.pkl")
joblib.dump(vectorizer, "static/models/vectorizer.pkl")