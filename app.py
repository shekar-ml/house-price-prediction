import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open('RidgeModel.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html', locations=data['location'].unique())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_df = pd.DataFrame([[
            request.form.get('location'),
            int(request.form.get('bhk')),
            int(request.form.get('bath')),
            float(request.form.get('square_feet'))
        ]], columns=['location', 'bhk', 'bath', 'total_sqft'])
        prediction = pipe.predict(input_df)[0] * 100000
        return str(np.round(prediction, 2))
    except Exception as e:
        return f"Prediction failed: {str(e)}"
    
if __name__ == "__main__":
    app.run(debug=True)