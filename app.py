import os
import pandas as pd
import statsmodels.api as sm
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the saved SARIMAX model from the pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the request contains a file
        if 'file' not in request.files:
            return jsonify({'error': 'No file found'})

        file = request.files['file']

        # Check if the file is a CSV file
        if file.filename == '' or not file.filename.endswith('.csv'):
            return jsonify({'error': 'Invalid file format. Please upload a CSV file'})

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv('X.csv', parse_dates=['Date'], index_col='Date')

        # Extract the exogenous variables from the DataFrame
        # exog = df[['exog_var1', 'exog_var2', ...]]  # List of your exogenous variables

        # Make predictions with the SARIMAX model
        predictions = model.predict(start= df.index[0], end = df.index[-1]
                                    , exog=df)

        # Convert predictions to a JSON response
        result = {'predictions': predictions.tolist()}

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Use the PORT environment variable for Heroku deployment
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
