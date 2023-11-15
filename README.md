# Machine Learning Spam Comment Detector

## Description
This repository contains the code for a logistic regression machine learning model trained to classify comments as spam or legitimate. Users can interact with the model through a Flask web interface and view insightful data visualizations crafted with Matplotlib. An interactive demo of this application can be accessed [here](http://techsmash.pythonanywhere.com/).

## Features
- **Logistic Regression Model**: Highly accurate in classifying comments. Trained from YouTube spam comment dataset.
- **Flask Web Interface**: Provides an interactive platform for users to demo the model.
- **Data Visualizations**: Uses Matplotlib to visually represent model insights and dataset features.

## Installation and Setup
1. **Clone the Repository**: Clone or download this repository to your local machine to get started.
   
```bash
git clone https://github.com/AndyAnderson8/Spam-Comment-Detector.git
cd Spam-Comment-Detector
```

2. **Install Required Packages**: Make sure you have Python and pip installed. Then, install the necessary packages:
   
```bash
pip install -r requirements.txt
```

### First Time Setup
If this is your first time running the application, execute the `init.py` script. This script will first set up the logistic regression model, then generate necessary data visualizations, and finally launch the Flask application.

```bash
python init.py
```

### Regular Use
After the initial setup, you can directly run the Flask application without going through the initial setup process.

```bash
python app.py
```

After running the above command, the Flask app will start, and you can access it at `http://127.0.0.1:5000/` in your web browser. From there, you can enter your own comments to test the model's classification, as well as view related visualizations for better insight.

## License
[MIT](https://github.com/AndyAnderson8/Spam-Comment-Detector/blob/main/LICENSE.txt)
