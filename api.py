from flask import Flask, request, render_template
import pickle
import pandas as pd
from car_data_prep import prepare_data
import werkzeug

app = Flask(__name__)

# Load the trained model
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():


    try:
        # Receiving data from the form
        input_data = {
            'manufactor': request.form['manufactor'],
            'Year': request.form['year'],
            'model': request.form['model'],
            'Hand': request.form['hand'],
            'Gear': request.form['gear'],
            'capacity_Engine': request.form['capacity_Engine'],
            'Engine_type': request.form['engine_type'],
            'Prev_ownership': request.form['prev_ownership'],
            'Curr_ownership': request.form['curr_ownership'],
            'City': request.form['city'],
            'Color': request.form['Color'],
            'Km': request.form['Km'],
            'Test': pd.NA,
            'Area': pd.NA,
            'Pic_num': pd.NA,
            'Cre_date': pd.NA,
            'Repub_date': pd.NA,
            'Description': 'Description',
            'Supply_score': 0
        }
    except werkzeug.exceptions.BadRequestKeyError as e:
        print(f"Missing key: {e.args[0]}")
        return render_template('index.html', prediction_text="Error: Missing data.")

    input_df = pd.DataFrame([input_data])

    # Data processing
    processed_input = prepare_data(input_df, fit=False)
    prediction = model.predict(processed_input)[0]


    return render_template('index.html', prediction_text=f'המחיר החיזוי הוא: {prediction:.2f} ש"ח')

if __name__ == '__main__':
    app.run(debug=True)
