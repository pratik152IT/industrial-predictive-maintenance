# Industrial Predictive Maintenance Microservice

## 📌 Project Overview
This project is an end-to-end MLOps solution designed to predict industrial machine failures before they occur, drastically reducing factory downtime and maintenance costs. 

Using the AI4I 2020 Predictive Maintenance Dataset, we developed a hybrid workflow: training a high-performance machine learning classifier in a cloud-based pipeline, and serving the model locally via a lightweight, production-ready REST API gateway.

## 🛠️ Tech Stack & Architecture
- **Data Pipeline & Training:** Python, Google Colab, Pandas, Scikit-Learn (Random Forest Classifier), Joblib
- **Production API Serving:** FastAPI, Pydantic, Uvicorn
- **Environment Isolation & Deployment:** Python Virtual Environments (`venv`), Docker

## 📂 Project Structure
```text
industrial-predictive-maintenance/
├── models/
│   ├── model.pkl            # Trained Random Forest Classifier
│   └── scaler.pkl           # Trained StandardScaler artifact
├── app.py                   # FastAPI production application logic
├── Dockerfile               # Deployment blueprint for containerization
└── requirements.txt         # Project dependencies