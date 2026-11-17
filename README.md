## EMIPredict AI — Intelligent Financial Risk Assessment Platform

A machine-learning-powered financial risk assessment platform that predicts a user's maximum affordable monthly EMI and classifies their EMI eligibility/risk profile based on demographic, employment, income, expense, credit, savings, and requested-loan information.

The project combines data preprocessing, exploratory data analysis, feature engineering, supervised machine learning, model artifact management, FastAPI inference services, and a React/Vite frontend into a production-oriented application.

* Domain: FinTech / Banking
* Primary Tasks: Regression + Classification
* Dataset: 400,000 financial records
* Backend: Python + FastAPI
* Frontend: React + TypeScript + Vite
* ML Tracking: MLflow
* Testing: Pytest

# Project Overview

EMIPredict AI is designed to help evaluate an individual's financial capacity before taking a loan.

The platform addresses two related machine-learning problems:

1. Regression: Predict the maximum monthly EMI a customer can reasonably afford.
2. Classification: Classify the customer's EMI eligibility into:
    * Eligible
    * High_Risk
    * Not_Eligible

The original project specification defines a financial risk assessment platform using 400,000 financial records, advanced feature engineering, classification and regression models, and MLflow experiment tracking.

The production implementation separates:

Data
  ↓
Data Quality / Preprocessing
  ↓
Feature Engineering
  ↓
Model Training
  ↓
Model Evaluation
  ↓
Persisted Model Artifacts
  ↓
FastAPI Inference Service
  ↓
React Frontend
  ↓
User Prediction

# Problem Statement

Loan applicants often do not know how much EMI they can safely afford.

A simple EMI calculator only answers:

`"What will my EMI be for this loan?"`

It does not answer:

`"Is this EMI financially appropriate for me?"`

EMIPredict AI attempts to answer both questions by considering the applicant's broader financial profile.

The project specification identifies the core objective as building a financial risk assessment platform that integrates machine learning and MLflow to provide interactive EMI prediction.

# Key Objectives
The platform is designed to:

* Predict maximum affordable monthly EMI.
* Assess EMI eligibility.
* Estimate probabilities for the three eligibility classes.
* Analyze income and expenditure patterns.
* Consider existing EMI obligations.
* Consider credit score and financial reserves.
* Engineer financial ratios from raw variables.
* Provide a production inference API.
* Provide an interactive web interface.
* Persist trained models as reusable artifacts.
* Validate incoming prediction data.
* Provide automated unit tests.
* Track ML experiments using MLflow.

The original specification calls for classification and regression solutions, feature engineering, model comparison, MLflow tracking, and a production web application.

# How the System Works

A user enters their financial information through the frontend.

The frontend collects information such as:

Personal Information
        ↓
Employment & Income
        ↓
Monthly Expenses
        ↓
Existing Commitments
        ↓
Requested Loan

The backend converts these values into the feature schema expected by the production models.

The inference service then:

1. Loads the persisted regression model.
2. Loads the persisted classification model.
3. Validates the feature schema.
4. Runs regression inference.
5. Runs classification inference.
6. Obtains classification probabilities.
7. Applies the minimum EMI business constraint.
8. Returns the prediction to the API.
9. The frontend displays the result.

The production inference service intentionally performs inference only. Model training is kept outside the inference path.


# System Architecture

                    ┌─────────────────────┐
                    │      User           │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ React + TypeScript  │
                    │      Frontend       │
                    └──────────┬──────────┘
                               │ HTTP
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │     REST API        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Inference Service   │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
          ┌─────────────────┐   ┌──────────────────┐
          │ Regression      │   │ Classification   │
          │ Model           │   │ Model            │
          └────────┬────────┘   └────────┬─────────┘
                   │                     │
                   └──────────┬──────────┘
                              ▼
                    ┌─────────────────────┐
                    │ Prediction Response │
                    └─────────────────────┘

The broader project architecture specified in the project brief consists of data, processing, model, application, and deployment layers.


# Machine Learning Problems

1. Regression

The regression model predicts:

`max_monthly_emi`

This represents the predicted maximum monthly EMI capacity.

The project specification defines this as a continuous target representing the maximum safe monthly EMI amount.

The production service additionally enforces:

`minimum predicted EMI = ₹500`

Therefore:

`predicted_emi = max(500, model_prediction)`

2. Classification

The classification model predicts:

`emi_eligibility`

with three possible classes:

`Eligible`
`High_Risk`
`Not_Eligible`

The project specification describes these classes as:

Class	        Meaning
`Eligible`	    Lower-risk profile with comfortable EMI affordability
`High_Risk`	    Marginal case requiring additional consideration
`Not_Eligible`	High-risk profile where the requested EMI is not recommended

The API also returns probabilities for all three classes.

Example:

`{
  "Eligible": 0.10,
  "High_Risk": 0.05,
  "Not_Eligible": 0.85
}`

These probabilities should be interpreted as model outputs, not as guaranteed probabilities of loan repayment or default.

# Input Features

The original dataset contains 22 primary input variables.

Personal Information
Feature 	          Description
age	                  Applicant age
gender	              Applicant gender
marital_status	      Marital status
education             Educational qualification

Employment and Income
Feature	                Description
monthly_salary	        Monthly gross salary
employment_type	        Employment category
years_of_employment	    Employment duration
company_type	        Company/organization type


Housing and Family
Feature	            Description
house_type	        Housing status
monthly_rent	    Monthly rental expense
family_size	        Household size
dependents	        Number of dependents


Monthly Expenses
Feature	                Description
school_fees	            School-related expenses
college_fees	        College-related expenses
travel_expenses	        Monthly travel expenses
groceries_utilities	    Groceries and utility expenses
other_monthly_expenses	Other monthly expenses


Existing Financial Commitments
Feature	            Description
existing_loans	    Whether existing loans exist
current_emi_amount	Existing monthly EMI
credit_score	    Credit score
bank_balance	    Current bank balance
emergency_fund	    Emergency savings


Requested Loan
Feature	            Description
emi_scenario	    Loan/EMI scenario
requested_amount	Requested loan amount
requested_tenure	Requested repayment tenure

The dataset specification describes five EMI scenarios: E-commerce Shopping, Home Appliances, Vehicle, Personal Loan, and Education EMI.


# Engineered Features

The system creates additional financial features from the raw data.

These include:

`total_living_expenses`
`total_monthly_obligations`
`disposable_income`
`expense_to_income`
`existing_emi_to_income`
`obligation_to_income`
`bank_balance_to_income`
`emergency_fund_to_income`


Total Living Expenses

Calculated as:

rent
+ school fees
+ college fees
+ travel expenses
+ groceries/utilities
+ other expenses


Total Monthly Obligations

total_living_expenses
+ current_emi_amount


Disposable Income
monthly_salary
- total_monthly_obligations


Expense-to-Income Ratio
`total_living_expenses / monthly_salary`


Existing EMI-to-Income Ratio
`current_emi_amount / monthly_salary`


Obligation-to-Income Ratio
`total_monthly_obligations / monthly_salary`


Bank Balance-to-Income Ratio
`bank_balance / monthly_salary`


Emergency Fund-to-Income Ratio
`emergency_fund / monthly_salary`

The project specification explicitly identifies debt-to-income, expense-to-income, affordability ratios, and other derived financial features as part of the feature-engineering stage.

# Technology Stack
Backend
* Python 3.12
* FastAPI
* Pydantic
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Uvicorn

Machine Learning
* Classification
* Regression
* Feature Engineering
* Model Evaluation
* Model Artifact Persistence
* MLflow

The project specification proposes Logistic Regression, Random Forest, and XGBoost for classification and Linear Regression, Random Forest, and XGBoost for regression.

# Frontend
* React 19
* TypeScript
* Vite
* CSS
* Fetch API
* Testing
* Pytest
* FastAPI TestClient
* Development Tools
* Git
* GitHub
* Virtual environment
* npm

# Project Structure
EMI/
│
├── artifacts/
│   ├── classification/
│   │   ├── metadata.json
│   │   └── model.joblib
│   │
│   └── regression/
│       ├── metadata.json
│       └── model.joblib
│
├── data/
│   └── raw/
│       └── emi_prediction_dataset.csv
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── assets/
│   │   ├── App.tsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.tsx
│   │
│   ├── package.json
│   ├── package-lock.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── .gitignore
│
├── notebooks/
│   └── 01_data_quality.ipynb
│
├── scripts/
│   ├── build_features.py
│   └── build_model_artifacts.py
│
├── src/
│   └── emipredict/
│       ├── api/
│       │   ├── __init__.py
│       │   ├── dependencies.py
│       │   ├── main.py
│       │   └── schemas.py
│       │
│       ├── analysis/
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py
│       │
│       ├── data/
│       │   ├── cleaner.py
│       │   ├── loader.py
│       │   └── split.py
│       │
│       ├── features/
│       │   └── features.py
│       │
│       ├── inference/
│       │   ├── __init__.py
│       │   └── service.py
│       │
│       └── models/
│           ├── __init__.py
│           ├── artifacts.py
│           ├── classification.py
│           └── regression.py
│
├── tests/
│   └── unit/
│       ├── test_api.py
│       ├── test_artifacts.py
│       ├── test_classification.py
│       ├── test_config.py
│       ├── test_data_cleaning.py
│       ├── test_data_validation.py
│       ├── test_eda.py
│       ├── test_features.py
│       ├── test_inference.py
│       ├── test_regression.py
│       └── test_split.py
│
├── pyproject.toml
├── requirements.txt
└── README.md

`frontend/node_modules/, Python virtual-environment files, caches, and other generated development files should remain excluded through .gitignore.`


# Backend API

The backend exposes a REST API using FastAPI.

* Health Check
`GET /health`

Example response:

{
  "status": "healthy",
  "service": "emi-prediction-api"
}


* Model Information
`GET /model-info`

Returns:

* Feature count
* Feature schema
* Model metadata
* Minimum EMI constraint

Example structure:

{
  "feature_count": 33,
  "feature_schema": [],
  "models": {},
  "min_monthly_emi": 500
}

The production feature schema includes the original model features plus engineered financial ratios.

# Single Prediction
`POST /predict`

The endpoint accepts one feature record.

{
  "features": {
    "age": 24,
    "gender": "Male",
    "marital_status": "Single",
    "education": "Graduate"
  }
}

The actual request must contain all features required by the production model.

# Batch Prediction
`POST /predict/batch`

Accepts multiple feature records.

The API supports up to 1,000 records per batch request according to the request schema.

API Response

A single prediction contains:

{
  "predicted_max_monthly_emi": 12000.0,
  "emi_eligibility": "Eligible",
  "probabilities": {
    "Eligible": 0.87,
    "High_Risk": 0.08,
    "Not_Eligible": 0.05
  }
}

The API validates that the three probabilities sum to approximately 1.0.

# Frontend

The frontend is implemented using:

React
+
TypeScript
+
Vite
+
CSS

The frontend provides a financial application form covering:

1. Personal information
2. Employment and income
3. Monthly expenses
4. Existing commitments
5. Requested loan

The current form implementation contains fields such as age, gender, marital status, education, family size, dependents, employment information, salary, expenses, existing EMI, credit score, bank balance, emergency fund, EMI scenario, requested amount, and tenure.

The frontend is designed to communicate with the FastAPI backend and present the resulting EMI and eligibility information to the user.


# Model Artifacts

Trained production models are stored separately from application source code:

artifacts/
├── classification/
│   ├── metadata.json
│   └── model.joblib
│
└── regression/
    ├── metadata.json
    └── model.joblib

This separation is intentional.

The API does not train models during prediction.

Instead:

Training
   ↓
Model artifact
   ↓
Artifact manager
   ↓
Inference service
   ↓
Prediction

This makes the inference layer deterministic and suitable for deployment.

# Installation
1. Clone the Repository
`git clone <repository-url>`
`cd EMI`


2. Create Python Virtual Environment

Windows:

`python -m venv .venv`

Activate it:

`.\.venv\Scripts\Activate.ps1`


3. Install Python Dependencies
`pip install -r requirements.txt`


4. Install Frontend Dependencies
`cd frontend`
`npm install`


# Running the Backend

From the project root:

`python -m uvicorn emipredict.api.main:app --host 127.0.0.1 --port 8000`

Backend:

`http://127.0.0.1:8000`

Swagger API documentation:

`http://127.0.0.1:8000/docs`

Health check:

`http://127.0.0.1:8000/health`



# Running the Frontend

Open another terminal:

`cd E:\MACHINE LEARNING\EMI\frontend`
`npm run dev`

Vite will provide a local URL, normally:

`http://localhost:5173/`

The backend and frontend need to be running simultaneously for end-to-end prediction.


# Using the Application
1. Start the FastAPI backend.
2. Start the Vite frontend.
3. Open the frontend URL.
4. Enter applicant information.
5. Enter requested loan information.
6. Submit the prediction.
7. The frontend sends the request to the API.
8. The API validates the input.
9. The inference service runs both models.
10. The prediction result is returned.
11. The frontend displays the result.


# API Examples

Health Check

PowerShell:

* Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/health" `
  -Method GET

Model Information
* Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/model-info" `
  -Method GET


# Prediction

A complete request should follow the production feature schema returned by:

`GET /model-info`

Example structure:

{
  "features": {
    "age": 30,
    "gender": "Male",
    "marital_status": "Single",
    "education": "Graduate",
    "monthly_salary": 75000,
    "employment_type": "Private",
    "years_of_employment": 5,
    "company_type": "MNC",
    "house_type": "Rented",
    "monthly_rent": 15000,
    "family_size": 3,
    "dependents": 1,
    "school_fees": 5000,
    "college_fees": 0,
    "travel_expenses": 3000,
    "groceries_utilities": 10000,
    "other_monthly_expenses": 3000,
    "existing_loans": "No",
    "current_emi_amount": 0,
    "credit_score": 750,
    "bank_balance": 200000,
    "emergency_fund": 50000,
    "emi_scenario": "Education EMI",
    "requested_amount": 200000,
    "requested_tenure": 60,
    "total_living_expenses": 36000,
    "total_monthly_obligations": 36000,
    "disposable_income": 39000,
    "expense_to_income": 0.48,
    "existing_emi_to_income": 0,
    "obligation_to_income": 0.48,
    "bank_balance_to_income": 2.67,
    "emergency_fund_to_income": 0.67
  }
}

The exact categorical values must match the values supported by the trained production pipeline.

# Testing

The project contains unit tests covering:

* API behavior
* Model artifacts
* Classification
* Configuration
* Data cleaning
* Data validation
* EDA
* Feature engineering
* Inference
* Regression
* Dataset splitting

Run all unit tests:

`python -m pytest tests/unit`

A successful test run currently reports:

160 passed, 4 warnings

Warnings should still be reviewed periodically; a passing test suite does not mean warnings should automatically be ignored.



# Frontend Production Build

From the frontend directory:

`npm run build`

Successful output should contain:

`✓ built in ...`

The generated production files are placed under:

`frontend/dist/`

The build command performs:

TypeScript compilation
        ↓
Vite production build
        ↓
frontend/dist/


# Frontend Development

Start the development server:

`npm run dev`

Lint the frontend:

`npm run lint`

Build for production:

`npm run build`

Preview the production build:

`npm run preview`


# Feature Engineering

Feature engineering is separated into its own processing component.

The objective is to transform cleaned financial records into financially meaningful derived variables.

The feature engineering layer does not:

* Load raw data from disk.
* Clean malformed data.
* Train models.
* Perform predictions.
* Modify target variables.
* Perform train/test splitting.

This separation prevents the inference layer from becoming responsible for the entire ML pipeline.


# Inference Service

The central production inference component is:

`EMIInferenceService`

It is responsible for:

* Loading regression artifacts.
* Loading classification artifacts.
* Validating model compatibility.
* Validating input feature schemas.
* Running regression inference.
* Running classification inference.
* Returning classification probabilities.
* Supporting batch inference.
* Returning model metadata.

The service verifies that both models use compatible feature schemas.

It also validates the expected production targets:

Regression:
`max_monthly_emi`

Classification:
`emi_eligibility`


# Business Logic

The system applies a minimum EMI constraint:

`Minimum monthly EMI = ₹500`

If the regression model produces a value below ₹500:

`Model prediction → ₹300`
`Production result → ₹500`

This is a business constraint, not a change to the underlying trained model.

This distinction is important when interpreting model performance.


# Eligibility Interpretation

The three classification results should be treated as risk categories generated by the trained model.

Eligible

Generally represents a profile where the model considers the financial capacity compatible with the evaluated EMI scenario.

Potential contributing variables can include:

* Higher disposable income
* Lower existing obligations
* Stronger credit profile
* Stable employment
* Lower expense-to-income ratio
* Greater financial reserves

However, the model prediction should not be interpreted as proof that a customer will repay a loan.

High Risk

Represents a marginal profile.

The model may identify financial pressure such as:

* Higher obligations relative to income
* Existing EMI burden
* Lower disposable income
* Weaker financial reserves
* Other learned risk patterns

A High_Risk prediction should be interpreted as requiring additional financial evaluation.

Not Eligible

Represents the highest-risk classification according to the trained model.

It may be associated with a combination of:

* High expense burden
* High existing obligations
* Low disposable income
* Weak credit characteristics
* Insufficient financial reserves
* Loan scenario characteristics

Importantly, these are interpretive factors, not guaranteed causal explanations of an individual prediction. A tree-based or ensemble model can use interactions between variables that are not captured by a simple rule such as "credit score below X means Not Eligible."

# Data Validation

The API uses Pydantic models to validate request and response structures.

The prediction request rejects unexpected fields.

The probability response validates:

`0 ≤ probability ≤ 1`

and:

Eligible
+ High_Risk
+ Not_Eligible
≈ 1.0

The inference service also checks:

* Required features
* Extra features
* Feature ordering
* Empty inputs
* Number of records
* Model compatibility

This prevents a common production ML problem: sending a structurally valid request with an incorrect feature schema.

# Error Handling

The API distinguishes between controlled inference errors and unexpected server errors.

Inference validation failures return:

`HTTP 422`

with a structured response such as:

{
  "detail": "...",
  "error_type": "inference_validation_error"
}

Unexpected application failures return:

`HTTP 500`

without exposing internal implementation details.

# Request Tracking and Logging

The API generates or accepts an `X-Request-ID`.

Example:

`X-Request-ID: 550e8400-e29b-41d4-a716-446655440000`

The response includes the same request ID.

The middleware records:

* Request ID
* HTTP method
* Request path
* HTTP status
* Request duration

This is useful for debugging production requests.

# MLflow

MLflow is intended to provide experiment tracking and model management.

The project specification calls for tracking:

* Model parameters
* Hyperparameters
* Evaluation metrics
* Model artifacts
* Model versions

and comparing model performance before selecting production models.

For classification, the planned evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

For regression:

* RMSE
* MAE
* R²
* MAPE

These metrics are explicitly specified in the project requirements.

# Model Selection

The intended development process compares multiple models rather than assuming one algorithm is automatically best.

Classification

Potential models:

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier
* Regression

Potential models:

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor

The project requirements specify a minimum of three models for each task and selection of the best-performing model for deployment.

# Data Pipeline

The overall ML pipeline is:

Raw Dataset
     │
     ▼
Data Loading
     │
     ▼
Data Quality Assessment
     │
     ▼
Cleaning & Validation
     │
     ▼
Train / Validation / Test Split
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Feature Engineering
     │
     ▼
Categorical Encoding
     │
     ▼
Numerical Processing
     │
     ▼
Model Training
     │
     ├──────────────┐
     ▼              ▼
Classification    Regression
     │              │
     └──────┬───────┘
            ▼
       Model Evaluation
            │
            ▼
       Model Selection
            │
            ▼
       Model Artifacts
            │
            ▼
      Production API

# Important ML/Data Considerations
Model Prediction ≠ Financial Advice

The application is a machine-learning prediction system.

Its output should not be treated as:

* Guaranteed loan approval
* Guaranteed loan rejection
* Guaranteed repayment probability
* Professional financial advice
* A bank's actual underwriting decision

Actual lending decisions can involve additional information and institutional policies.

Classification Probability ≠ Default Probability

For example:

`Not_Eligible: 85%`

does not necessarily mean:

"There is an 85% probability this applicant will default."

It means the classification model assigned approximately 85% probability to the `Not_Eligible` class under its learned classification framework.

This distinction is important when explaining predictions to users.

# Repository Hygiene

The repository should not contain:

.venv/
__pycache__/
*.pyc
.pytest_cache/
node_modules/
.env
mlruns/

These are development/runtime-generated files and should be excluded using `.gitignore`.

The frontend dependency directory:

`frontend/node_modules/`

must not be committed.

Instead, commit:

`frontend/package.json`
`frontend/package-lock.json`

and recreate dependencies with:

`npm install`

# Rebuilding Model Artifacts

The model-artifact build process is separated from inference.

The artifact-building script is:

`scripts/build_model_artifacts.py`

Its purpose is to create persisted production artifacts such as:

`artifacts/classification/model.joblib`
`artifacts/classification/metadata.json`

`artifacts/regression/model.joblib`
`artifacts/regression/metadata.json`

The inference service should consume these artifacts rather than retraining models on every API startup.

Production Separation

A major design principle of this project is separating training from serving.

Training environment
Dataset
→ preprocessing
→ feature engineering
→ model training
→ evaluation
→ model selection
→ artifact creation


Production environment
API request
→ validation
→ feature preparation
→ load artifacts
→ inference
→ response

This avoids unnecessary training during user requests and makes deployment easier to reason about.

Current Validation

The backend unit-test suite currently passes:

160 passed, 4 warnings

The frontend production build also completes successfully with:

tsc -b
+
vite build

The successful frontend build confirms that the current TypeScript code compiles and Vite can generate a production bundle.

# Known Development Issue: Failed to Fetch

If the frontend displays:

`Failed to fetch`

this generally indicates that the browser could not successfully communicate with the backend.

The two services must both be running:

Terminal 1
`python -m uvicorn emipredict.api.main:app --host 127.0.0.1 --port 8000`


Terminal 2
`cd frontend`
`npm run dev`

Then verify:

`http://127.0.0.1:8000/health`

before debugging the frontend prediction request.

# Future Improvements

Several improvements would make the system more robust.

1. Explainable Predictions

Instead of displaying only:

`Not Eligible — 85%`

the system could calculate model-specific feature contributions using techniques such as:

* SHAP
* permutation importance
* model-specific feature importance

This would allow explanations to be tied to the actual prediction rather than generic financial rules.

2. Requested Loan EMI

The frontend can separately calculate:

`Requested Loan EMI`

using:

* Principal
* Interest Rate
* Loan Tenure

This allows users to compare:

* Requested EMI
* vs.
* Predicted Maximum EMI

That is more useful than showing the model's maximum EMI alone.

3. Loan Interest Rate

The user-facing loan calculation can accept:

`Interest rate (%)`

and use it only for calculating the requested loan's EMI.

This should not automatically be added to the ML model unless the model was trained with interest-rate information. Adding a frontend field without retraining the model would otherwise create a misleading impression that the model uses that variable.

4. Loan Tenure in Years

The user interface can present:

`Loan tenure: 5 years`

while converting it internally to months:

`5 years × 12 = 60 months`

if the production model expects requested_tenure in months.

The original project specification defines requested tenure in months.

5. Better Categorical Normalization

The dataset contains categorical inconsistencies such as:

Male
male
MALE
M

and:

Female
female
FEMALE
F

A production preprocessing layer should normalize these consistently before inference.

6. Automated CI

GitHub Actions could automatically run:

`pytest tests/unit`
`npm run build`
`npm run lint`

for every pull request.

# Future Deployment Architecture

A possible production architecture is:

                  Internet
                     │
                     ▼
             ┌───────────────┐
             │ React Frontend│
             └───────┬───────┘
                     │ HTTPS
                     ▼
             ┌───────────────┐
             │ FastAPI       │
             │ Backend       │
             └───────┬───────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
 ┌─────────────────┐   ┌──────────────────┐
 │ Regression      │   │ Classification   │
 │ Artifact        │   │ Artifact         │
 └─────────────────┘   └──────────────────┘

The original project specification proposed cloud deployment and automated deployment through GitHub.

# Project Status
Component	                        Status
Dataset processing	                Implemented
Data quality checks	                Implemented
Feature engineering	                Implemented
Train/test splitting	            Implemented
Classification pipeline	            Implemented
Regression pipeline	                Implemented
Model artifacts	                    Implemented
Inference service	                Implemented
FastAPI API	                        Implemented
API validation                  	Implemented
Unit tests	                        Passing
React frontend	                    Implemented
Vite production build	            Passing
Requested-loan EMI calculation	    Implemented
Production deployment	            Not yet documented as completed


# Development Commands
Backend
<!-- Activate environment -->
`.\.venv\Scripts\Activate.ps1`

<!-- Run tests -->
`python -m pytest tests/unit`

<!-- Start API -->
`python -m uvicorn emipredict.api.main:app --host 127.0.0.1 --port 8000`


# Frontend
`cd frontend`

<!--  Install dependencies -->
`npm install`

<!-- Development server -->
`npm run dev`

<!-- Production build -->
`npm run build`

<!-- Lint -->
`npm run lint`

<!-- Preview production build -->
`npm run preview`


# Final Architecture Summary

EMIPredict AI is structured around a separation of responsibilities:

                 EMIPredict AI
                      │
       ┌──────────────┼──────────────┐
       │              │              │
       ▼              ▼              ▼
     Data           Models       Application
       │              │              │
       ▼              ▼              ▼
 Cleaning       Classification    FastAPI
 Validation     Regression         React
 Features       Artifacts          TypeScript
       │              │              │
       └──────────────┼──────────────┘
                      ▼
               EMI Risk Assessment

The central value of the platform is not simply calculating an EMI. It combines financial capacity features, machine-learning regression, classification, probability outputs, and a user-facing application to provide a broader assessment of the requested loan scenario.

The project requirements identify the intended end goal as a production-ready financial platform capable of real-time prediction and supporting data-driven lending decisions.