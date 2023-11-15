import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import joblib
import itertools

#Step 2

clf = joblib.load("static/models/spam_classifier.pkl")
vectorizer = joblib.load("static/models/vectorizer.pkl")

df = pd.read_csv("static/dataset.csv")
X = df["CONTENT"] #comments
y = df["CLASS"] #spam classification

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_test_vectorized = vectorizer.transform(X_test)

def plot_roc_curve(y_test, prediction_probs):
  fpr, tpr, _ = roc_curve(y_test, prediction_probs[:, 1])
  roc_auc = auc(fpr, tpr)
  plt.figure()
  plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC Curve (Area = %0.2f)" % roc_auc)
  plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("Receiver Operating Characteristic Curve")
  plt.legend(loc="lower right")
  plt.savefig("static/visualizations/roc_curve.png")
  print("ROC Curve visualization created")

def plot_confusion_matrix(y_test, predictions):
  cm = confusion_matrix(y_test, predictions)
  plt.figure(figsize=(7,7))
  plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
  plt.title("Confusion Matrix")
  plt.colorbar()
  
  #numbers for each quadrent
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
      horizontalalignment="center",
      color="white" if cm[i, j] > (cm.max() / 2.) else "black")

  tick_marks = np.arange(2)
  plt.xticks(tick_marks, ["Not Spam", "Spam"], rotation=45)
  plt.yticks(tick_marks, ["Not Spam", "Spam"])
  plt.tight_layout(pad=2.5) #needs more padding, dont remove
  plt.ylabel("True Label")
  plt.xlabel("Predicted Label")
  plt.savefig("static/visualizations/confusion_matrix.png")
  print("Confusion Matrix visualization created")

def plot_feature_importance(vectorizer, clf):
  feature_names = vectorizer.get_feature_names_out()
  coef = clf.coef_.ravel()
  top_positive_coefficients = np.argsort(coef)[-20:]
  top_negative_coefficients = np.argsort(coef)[:20]
  top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
  
  plt.figure(figsize=(15, 7))
  colors = ["green" if c < 0 else "red" for c in coef[top_coefficients]]
  bars = plt.bar(np.arange(2 * 20), coef[top_coefficients], color=colors)
  
  plt.text(10, max(coef[top_coefficients]) - .5, "Not Spam", horizontalalignment="center", color="green", fontsize=14)
  plt.text(30, max(coef[top_coefficients]) - .5, "Spam", horizontalalignment="center", color="red", fontsize=14)

  feature_names = np.array(feature_names) #words in comment
  plt.xticks(np.arange(1, 1 + 2 * 20), feature_names[top_coefficients], rotation=60, ha="right")
  plt.title("Top Used Features Indicating Spam or Not Spam")
  plt.tight_layout(pad=2.5)  #extra padding, dont remove
  plt.savefig("static/visualizations/feature_importance.png")
  print("Feature Importance visualization created")

#create visualizations
predictions = clf.predict(X_test_vectorized)
prediction_probs = clf.predict_proba(X_test_vectorized)

plot_confusion_matrix(y_test, predictions)
plot_roc_curve(y_test, prediction_probs)
plot_feature_importance(vectorizer, clf)