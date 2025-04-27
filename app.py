import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import pickle
import os
from model import GymRecommendationModel

app = Flask(__name__)

# Load the pickled GymRecommendationModel using a relative path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form inputs
    user_input = {
        'Sex': int(request.form['Sex']),
        'Age': float(request.form['Age']),
        'Height': float(request.form['Height']),
        'Weight': float(request.form['Weight']),
        'Hypertension': int(request.form['Hypertension']),
        'Diabetes': int(request.form['Diabetes']),
        'BMI': float(request.form['BMI']),
        'Level': int(request.form['Level']),
        'Fitness Goal': int(request.form['Fitness Goal']),
        'Fitness Type': int(request.form['Fitness Type'])
    }

    # Get recommendations
    recommendations = model.predict(user_input, top_n=3)

    # Pass recommendations to the template
    return render_template('index.html', 
                         recommendations=recommendations,
                         predict_text="Here are your personalized workout and diet recommendations:")

if __name__ == '__main__':
    # Use the PORT environment variable provided by Render, default to 5000 for local testing
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
