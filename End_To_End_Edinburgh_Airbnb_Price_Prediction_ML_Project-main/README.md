
# 🏡 End-to-End Airbnb Price Prediction Project  
### **A Full Production-Ready, MLOps-Enabled Machine Learning System**

This project implements a complete **end-to-end machine learning pipeline** to predict Airbnb listing prices based on property features.  
It demonstrates the full lifecycle of an ML system—from data preparation and model training to containerization, CI/CD automation, and deployment on AWS & Azure.

---

# 📌 **Project Overview**

The goal of this project is to build a **scalable, reproducible, and production-ready ML application** capable of predicting Airbnb property prices.  
The project includes:

- Data ingestion & preprocessing  
- Feature engineering  
- Model training (Regression)  
- Model persistence  
- REST-ready app (Flask/FastAPI-ready structure)  
- **Docker containerization**  
- **GitHub Actions CI/CD pipeline**  
- **AWS deployment (ECR + EC2 Self-Hosted Runner)**  
- **Azure ACR deployment option**  

This project is ideal for showcasing **ML Engineering, MLOps, DevOps, and Cloud deployment** skills.

---

# 🧱 **Architecture Overview**

```
                +---------------------+
                |   Raw Airbnb Data   |
                +----------+----------+
                           |
                           v
                +---------------------+
                | Data Preprocessing  |
                |  Cleaning, Encoding |
                +----------+----------+
                           |
                           v
                +---------------------+
                |  Model Training     |
                | (Regression Model)  |
                +----------+----------+
                           |
                           v
                +---------------------+
                |   Model Registry    |
                |  (Saved .pkl file)  |
                +----------+----------+
                           |
                           v
       +----------- Docker Container -------------+
       |    ML Model + Web App (Flask/Streamlit)  |
       +------------------------------------------+
                           |
                           v
     +-----------------------------+   +-----------------------------+
     |     AWS ECR (Container)    |   |  Azure ACR (Container)      |
     +---------------+-------------+   +--------------+--------------+
                     |                               |
                     v                               v
         +----------------------+       +-------------------------+
         | AWS EC2 Deployment   |       | Future Azure VM/K8s     |
         +----------------------+       +-------------------------+
```

---

# 🚀 **Features**

### ✔ Full ML Pipeline  
- Data loading and cleaning  
- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Model training & evaluation  
- Model serialization (`joblib`)

### ✔ Dockerized Application  
- Predictive service fully containerized  
- Fast boot time  
- Works identically in all environments  

### ✔ CI/CD with GitHub Actions  
- Automatic Docker builds  
- Automatic push to AWS ECR  
- Supports production-grade pipelines  

### ✔ AWS Deployment  
- IAM-secured workflow  
- EC2 self-hosted runner  
- ECR container hosting  

### ✔ Azure Deployment Option  
- Build → Login → Push to Azure Container Registry (ACR)

---

# 🐍 **Tech Stack**

| Domain | Technologies |
|--------|--------------|
| ML | Python, Pandas, NumPy, Scikit-learn |
| Deployment | Docker, Flask/Streamlit |
| DevOps | GitHub Actions |
| Cloud | AWS EC2, ECR, IAM, Azure ACR |
| MLOps | Model packaging, CI/CD automation |

---

# 📘 **Project Structure**

```
Airbnb-Price-Prediction/
│── data/
│── notebooks/
│── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│── model/
│   └── model.pkl
│── Dockerfile
│── requirements.txt
│── github/workflows/
│   └── deploy.yml
│── README.md
```

---

# ⚙️ **Setup Instructions**

---

## 🐳 **1. Install Docker on EC2**

Run these commands on Ubuntu EC2:

```bash
sudo apt-get update -y
sudo apt-get upgrade -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

---

## 🔐 **2. Create IAM user in AWS**

IAM permissions required:

- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonEC2FullAccess`
- `IAMReadOnlyAccess`

Save the access keys into GitHub Secrets.

---

## 🔑 **3. Add GitHub Secrets**

| Secret Name | Description |
|------------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM User Key |
| `AWS_SECRET_ACCESS_KEY` | IAM Secret |
| `AWS_REGION` | Example: `us-east-1` |
| `AWS_ECR_LOGIN_URI` | Ex: `566373416292.dkr.ecr.ap-south-1.amazonaws.com` |
| `ECR_REPOSITORY_NAME` | Ex: `airbnb-app` |

---

# ⚡ **4. Configure EC2 as GitHub Self-Hosted Runner**

GitHub → Repository → Settings → Actions → Runners →  
**“New self-hosted runner”**  

Follow instructions and run the commands on EC2.

---

# 🐳 **5. Docker Commands (Azure Deployment)**

### Build image:
```bash
docker build -t airbnbpricepredic.azurecr.io/airbnb_prediction:latest .
```

### Login to Azure:
```bash
docker login airbnbpricepredic.azurecr.io
```

### Push:
```bash
docker push airbnbpricepredic.azurecr.io/airbnb_prediction:latest
```

---

# 📈 **Model Overview**

The model predicts price using:

- Property type  
- Bedrooms & bathrooms  
- Location-based features  
- Host attributes  
- Guest capacity  
- Review & rating metrics  

Model Type: **Regression (RandomForest / XGBoost / Linear)**  
Evaluation Metrics:  
- RMSE  
- MAE  
- R² Score  

---

# 🎯 **What This Project Demonstrates**

This project showcases real-world ML engineering skills:

### ✔ Building ML models from scratch  
### ✔ Applying MLOps & DevOps tools  
### ✔ Cloud deployment (AWS & Azure)  
### ✔ Docker containerization  
### ✔ CI/CD pipeline automation  
### ✔ Scalable architecture thinking  

Perfect for roles such as:

- **Data Scientist**  
- **ML Engineer**  
- **MLOps Engineer**  
- **Data Engineer (ML-focused)**  

---

# 📝 **Future Improvements**

- Add FastAPI for real-time prediction API  
- Deploy on AWS ECS or Kubernetes  
- Add monitoring with Prometheus & Grafana  
- Use MLflow for model tracking  
- Add feature store (Feast)

---

# 👨‍💻 **Author**
Your Name  
Machine Learning & MLOps Enthusiast  

---

