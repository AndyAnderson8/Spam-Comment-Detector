from flask import Flask, request, render_template
import joblib

#Step 3

print("Starting flask app...")

app = Flask(__name__)

#load models
clf = joblib.load("static/models/spam_classifier.pkl") 
vectorizer = joblib.load("static/models/vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
  
  #initialize
  result = None
  result_class = None
  confidence = None

  if request.method == "POST": #form submit
    comment = request.form["comment"] #text input
    comment_vectorized = vectorizer.transform([comment])

    prediction = clf.predict(comment_vectorized)
    prediction_probabilities = clf.predict_proba(comment_vectorized)
    confidence = round(prediction_probabilities[0][list(clf.classes_).index(prediction[0])] * 100, 2)

    match prediction:
      case 0:
        result_class = "not-spam" #class needed to set color on webpage
        result = "Not Spam"
      case _:
        result_class = "spam"
        result = "Spam"

  return render_template("index.html", result=result, result_class=result_class, confidence=confidence)

@app.route("/visualizations") #visualizations page 
def visualizations():
    return render_template("visualizations.html")

if __name__ == "__main__":
  app.run(debug=True)