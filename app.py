from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = 'zephyr_secret_key_2025'

# Load model, scaler, and encoders
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

# Features required by the model
FEATURES = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges']

# Dropdown options for select fields (categorical/boolean)
SELECT_OPTIONS = {
    'gender': ['Male', 'Female'],
    'SeniorCitizen': ['0', '1'],
    'Partner': ['Yes', 'No'],
    'Dependents': ['Yes', 'No'],
    'PhoneService': ['Yes', 'No'],
    'MultipleLines': ['Yes', 'No', 'No phone service'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['Yes', 'No', 'No internet service'],
    'OnlineBackup': ['Yes', 'No', 'No internet service'],
    'DeviceProtection': ['Yes', 'No', 'No internet service'],
    'TechSupport': ['Yes', 'No', 'No internet service'],
    'StreamingTV': ['Yes', 'No', 'No internet service'],
    'StreamingMovies': ['Yes', 'No', 'No internet service'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['Yes', 'No'],
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
}

# Dummy user login
users = {
    'admin': 'admin123',
    'zephyr': 'zephyr2025'
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        if uname in users and users[uname] == pwd:
            session['username'] = uname
            return redirect(url_for('predict_form'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/predict_form')
def predict_form():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('predict.html', features=FEATURES, options=SELECT_OPTIONS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {}
        for feature in FEATURES:
            input_data[feature] = request.form[feature]

        df = pd.DataFrame([input_data])

        # Encode categorical columns
        for col in df.columns:
            if col in encoders:
                df[col] = encoders[col].transform(df[col])

        # Convert numeric fields to float
        df[['tenure', 'MonthlyCharges', 'TotalCharges']] = df[['tenure', 'MonthlyCharges', 'TotalCharges']].astype(float)

        # Scale numeric columns
        df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])

        # Predict
        prediction = model.predict(df)[0]
        result = "✅ Customer likely to churn." if prediction == 1 else "✅ Customer likely to stay."
    except Exception as e:
        result = f"❌ Error: {e}"

    return render_template('result.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
