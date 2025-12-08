# 🎬 NLP-Based Movie Recommendation System

A content-based movie recommender system built using **Python, NLP
(TF-IDF)**, cosine/sigmoid similarity, and deployed with **Streamlit**.\
This project analyzes movie descriptions and recommends top similar
movies using semantic similarity techniques.

------------------------------------------------------------------------

## 🚀 Overview

This project builds an intelligent movie recommendation engine using the
following steps:

-   Load movie metadata\
-   Vectorize movie plots using **TF-IDF**\
-   Compute similarity matrix using **Sigmoid Kernel**\
-   Build a recommender function to return top similar movies\
-   Provide an interactive **Streamlit app** for users

------------------------------------------------------------------------

## 🧠 How It Works

### 1️⃣ TF-IDF Vectorization

The movie overview text is transformed into numeric vectors.

### 2️⃣ Similarity Computation

A similarity matrix (sigmoid kernel) is created to compare movies.

### 3️⃣ Recommendation Engine

Given a movie title: - Retrieve its index\
- Sort similarity scores\
- Return top 10 similar movies

### 4️⃣ Streamlit App

Interactive UI for selecting a movie and viewing recommendations.

------------------------------------------------------------------------

## 📁 Project Structure

    Movie-Recommender/
    │── data2_for_app.csv
    │── dataframe_for_app.csv
    │── tfv_vec.pkl
    │── sig_kernel.pkl
    │── app.py
    │── README.md

------------------------------------------------------------------------

## 📦 Technologies Used

### Core Libraries

-   Pandas\
-   Joblib\
-   Scikit-learn\
-   Streamlit

### NLP Techniques

-   TF-IDF Vectorization\
-   Sigmoid Kernel Similarity\
-   Content-Based Recommendation

------------------------------------------------------------------------

## ▶️ How to Run the App

### Install Dependencies

``` bash
pip install pandas joblib streamlit scikit-learn
```

### Run Streamlit App

``` bash
streamlit run app.py
```

The app will open at:

    http://localhost:8501

------------------------------------------------------------------------

## 🎯 Usage

1.  Choose a movie from the dropdown menu\
2.  Click **Get Recommendations**\
3.  View the top 10 similar movies

------------------------------------------------------------------------

## ⭐ Why This Project Is Valuable

This recommender demonstrates core data science skills:

-   NLP feature engineering\
-   Machine learning similarity modeling\
-   Data cleaning + preprocessing\
-   Model serialization\
-   Web deployment with Streamlit

Perfect for **Data Science portfolio** scenarios.

------------------------------------------------------------------------

## 🔮 Future Enhancements

-   Support hybrid recommendations (content + collaborative)\
-   Deploy to cloud (Streamlit Cloud)

------------------------------------------------------------------------

