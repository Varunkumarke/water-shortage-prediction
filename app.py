
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, request


data = pd.read_csv('WATER.csv')


X = data[['Rainfall Patterns (mm)','Population Growth (%)','Water Consumption (liters/person/day)','Year']]
y = data['Risk of Water Shortage (%)']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

app = Flask(__name__, template_folder='templates')

client = MongoClient('mongodb://localhost:27017')
db = client['water']
collection = db['db']


model = LinearRegression()
model.fit(X_train, y_train)


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    rainfall_pattern = float(request.form['rainfall_pattern'])
    population_growth = float(request.form['population_growth'])
    water_consumption = float(request.form['water_consumption'])
    year = float(request.form['year'])
    

    
    prediction = model.predict([[rainfall_pattern, population_growth, water_consumption,year]])[0]

    user = {
        'rainfall_pattern':rainfall_pattern,
        'population_growth':population_growth,
        'water_consumption':water_consumption,
        'Year':year,
        'prediction':prediction
    }

    collection.insert_one(user)
    
    return render_template('result.html',user=user)

if __name__ == '__main__':
    app.run(debug=True)
