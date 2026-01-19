from flask import Flask, render_template, request
import pandas as pd
import pickle

#Load model and scaler
with open('mlp_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    prediction_proba = None

    if request.method == 'POST':
        geography = request.form['geography']
        gender = request.form['gender']
        age = float(request.form['age'])
        balance = float(request.form['balance'])
        credit_score = float(request.form['credit_score'])
        estimated_salary = float(request.form['estimated_salary'])
        tenure = int(request.form['tenure'])
        num_of_products = int(request.form['num_of_products'])
        has_cr_card = int(request.form['has_cr_card'])
        is_active_member = int(request.form['is_active_member'])

        #prepare input data as DataFrame
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary],
            'Geography_Germany': [1 if geography == 'Germany' else 0],
            'Geography_Spain': [1 if geography == 'Spain' else 0],
            'Gender_Male': [1 if gender == 'Male' else 0]
        })
        
        #Order columns same as training dataset
        expected_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                        'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                        'Geography_Germany', 'Geography_Spain', 'Gender_Male']
        
        input_data = input_data[expected_cols]
        
        #Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0][1]

    return render_template('index.html', prediction=prediction, prediction_proba=prediction_proba)
    

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port=3000, debug=True)
