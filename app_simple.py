import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, request

# Load data
data = pd.read_csv('WATER.csv')

# Prepare features and target
X = data[['Rainfall Patterns (mm)','Population Growth (%)','Water Consumption (liters/person/day)','Year']]
y = data['Risk of Water Shortage (%)']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Flask app
app = Flask(__name__, template_folder='templates')

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    rainfall_pattern = float(request.form['rainfall_pattern'])
    population_growth = float(request.form['population_growth'])
    water_consumption = float(request.form['water_consumption'])
    year = float(request.form['year'])
    
    # Make prediction
    prediction = model.predict([[rainfall_pattern, population_growth, water_consumption, year]])[0]

    user = {
        'rainfall_pattern': rainfall_pattern,
        'population_growth': population_growth,
        'water_consumption': water_consumption,
        'Year': year,
        'prediction': prediction
    }
    
    return render_template('result.html', user=user)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
