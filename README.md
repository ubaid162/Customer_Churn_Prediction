# 🧠 Customer Churn Prediction – Flask Web App

A machine-learning powered web application that predicts whether a telecom customer will **churn** (leave the service) or **stay**.  
Built using **Flask, Scikit-learn, Pandas, and Bootstrap**, this project demonstrates full-cycle deployment of a predictive model through an interactive web interface.

---

## 📌 Overview
The app helps telecom businesses identify at-risk customers by analyzing service details, payment methods, and usage patterns.  
It uses a **Random Forest Classifier** trained on the **Telco Customer Churn dataset** to deliver real-time churn insights.

---

## 🧰 Tech Stack
| Layer | Technology |
|-------|-------------|
| Frontend | HTML5, CSS3, Bootstrap 5, Jinja2 Templates |
| Backend | Flask (Python) |
| ML Libraries | Scikit-learn, Pandas, NumPy, Joblib |
| Dataset | Telco Customer Churn (Kaggle) |

---

## 📂 Project Structure
Customer_Churn_Prediction/
├── app.py # Flask backend & prediction routes
├── model.pkl # Trained Random Forest model
├── scaler.pkl # StandardScaler object
├── encoders.pkl # LabelEncoders for categorical variables
├── Telco-Customer-Churn.csv # Dataset used for training
├── templates/ # HTML templates
│ ├── home.html
│ ├── about.html
│ ├── login.html
│ ├── navbar.html
│ ├── predict.html
│ └── result.html
├── static/ # Images, background assets
├── requirements.txt
└── README.md



---

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
