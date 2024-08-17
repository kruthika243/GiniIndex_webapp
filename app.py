from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('gini_index_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input from the form
        population = float(request.form.get('population', 0))
        gdp = float(request.form.get('gdp', 0))
        gdp_per_capita = float(request.form.get('gdp_per_capita', 0))
        area = float(request.form.get('area', 0))
        country = request.form.get('country', '')
        income_group = request.form.get('income_group', '')

        # Collect percentile values
        percentiles = [float(request.form.get(f'p{i}', 0)) for i in range(1, 101)]
        
        # Prepare the input data
        data = {
            'Population': [population],
            'GDP': [gdp],
            'GDP_per_capita': [gdp_per_capita],
            'Area': [area],
            'Country': [country],
            'Income_group': [income_group],
            **{f'Percentile_{i}': [percentiles[i-1]] for i in range(1, 101)}
        }
        
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        
        # Preprocess data if needed
        df_scaled = scaler.transform(df)  # Adjust if using specific feature transformers

        # Make prediction
        prediction = model.predict(df_scaled)
        
        # Convert numpy array to scalar if needed
        if isinstance(prediction, np.ndarray):
            prediction = prediction[0]  # Get the first element if it's a single value
        
        return jsonify({'gini_index': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
