# 🧠 Stroke Prediction System  

An **end-to-end Machine Learning project** that predicts the likelihood of stroke occurrence based on patient health data. The pipeline includes **data ingestion, preprocessing, model training, evaluation, and deployment** via a Flask web application.  

---

## 📊 Dataset  
The project uses the **[Healthcare Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)**.  
- **Features:** Demographic and medical history (age, BMI, glucose level, hypertension, heart disease, smoking status, etc.).  
- **Target:** `stroke` (binary: 1 = stroke, 0 = no stroke).  
- **Challenge:** Highly imbalanced dataset → handled using **SMOTE (Synthetic Minority Oversampling Technique)**.  

---

## ⚙️ Features Implemented  

### 1. **Data Ingestion**  
- Reads raw CSV dataset and splits into stratified train/test sets.  
- Saves artifacts (`train.csv`, `test.csv`, `data.csv`) for reproducibility.  

### 2. **Data Transformation**  
- Preprocessing pipeline using **ColumnTransformer**:  
  - **Numerical:** Imputation (median), scaling.  
  - **Categorical:** One-Hot Encoding, Target-Guided Encoding, imputation, scaling.  
- Handled imbalance with **SMOTE**.  
- Saved preprocessing object (`preprocessor.pkl`) for deployment.  

### 3. **Model Training & Selection**  
- Trained multiple ML models:  
  - Random Forest, Logistic Regression, SVM, Decision Tree, KNN.  
- **Hyperparameter tuning with GridSearchCV**.  
- Selected best-performing model based on test accuracy.  
- Saved trained model (`model.pkl`).  

### 4. **Flask Deployment**  
- Interactive web form for users to input patient health data.  
- Custom prediction pipeline ensures consistent preprocessing at inference time.  
- Returns real-time stroke prediction.  

---

## 🚀 Tech Stack  
- **Languages & Libraries:** Python, Pandas, NumPy, Scikit-learn, Imbalanced-learn  
- **Modeling:** SMOTE, GridSearchCV, multiple ML classifiers  
- **Deployment:** Flask  
- **Other:** Logging, Exception Handling  



---

## 🔧 How to Run  

### 1. Clone the repository  
```bash
git clone 1haj/End_To_End_Stroke_Prediction_System
cd stroke-prediction

### 2. Create a virtual environment & install dependencies
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
pip install -r requirements.txt
### 3. Run the Flask app
python app.py
The app will be available at: http://127.0.0.1:5000/🚀
