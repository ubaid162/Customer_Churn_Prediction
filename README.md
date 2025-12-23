# Customer Churn Prediction System

## Overview
A machine learning web application that predicts whether a telecom customer is likely to churn (leave the service) based on their usage patterns and account details.

## Business Problem
Customer churn costs telecom companies millions in lost revenue. This system helps identify at-risk customers early so the business can take retention actions like special offers or personalized outreach.

## Tech Stack
- **Python**: Pandas, NumPy, Scikit-learn
- **Machine Learning**: Random Forest Classifier
- **Web Framework**: Flask
- **Frontend**: HTML, CSS, Bootstrap

## Features
- Real-time churn prediction based on customer data
- User-friendly web interface for inputting customer details
- Displays churn probability and risk level
- Visual indicators for high-risk customers

## Model Performance
- **Algorithm**: Random Forest Classifier
- **Accuracy**: 90%
- **Key Features**: Contract type, payment method, tenure, monthly charges
- **Validation**: Train-test split (80-20)

## Key Insights from Analysis
- Month-to-month contract customers have highest churn rate
- Electronic payment methods correlate with higher churn
- Customers with tenure < 6 months are high risk
- Senior citizens show slightly higher churn tendency

## Dataset
Telecom customer dataset containing:
- Customer demographics (gender, age, partner status)
- Account information (contract type, payment method, tenure)
- Service usage (internet service, phone service, streaming)
- Billing details (monthly charges, total charges)

## How to Run
```bash
# Install dependencies
pip install pandas numpy scikit-learn flask

# Run the application
python app.py

# Open browser and go to
http://localhost:5000
```


## Future Improvements
- Add feature importance visualization
- Implement model retraining pipeline
- Add bulk prediction for multiple customers
## ⚙️ How It Works
1. **Home Page** – Welcome and intro section.  
2. **Login Page** – User authentication.  
3. **Prediction Form** – Input customer details.  
4. **Model Prediction** – Random Forest model predicts churn or stay.  
5. **Result Page** – Displays prediction result instantly.

---


Login Credentials
* Use one of these to test the app:
  * Username: admin → Password: admin123
  * Username: zephyr → Password: zephyr2025


 Future Enhancements
* Add data visualization dashboard (Plotly)
* Integrate database for storing predictions
* Implement XGBoost / Neural Network models
* Deploy on AWS or Render
