# 📱 Social Media Addiction Predictor

## 📖 Project Overview

This project is a comprehensive machine learning application designed to predict the impact of social media usage on students. It analyzes various personal, behavioral, and health-related factors to generate three key predictions:

1.  **Classification:** Predicts whether social media usage is likely to affect a student's academic performance (Yes/No).
2.  **Regression:** Estimates a student's social media addiction severity on a scale of 1 to 10.
3.  **Clustering:** Groups the user into a specific behavioral cluster (e.g., "Healthy User," "At-Risk User").

The project includes a Jupyter Notebook for data analysis and model training, and a user-friendly web interface built with Streamlit.

---

## ✨ Features

-   **Interactive Web Interface:** A clean and intuitive UI built with Streamlit for easy user input.
-   **Multi-Model Prediction:** Utilizes an ensemble of models (Decision Tree, Naive Bayes, Neural Network) for robust classification.
-   **Addiction Score Regression:** A Linear Regression model provides a quantitative addiction score.
-   **User Segmentation:** A K-Means model clusters users into distinct behavioral groups.
-   **Dynamic Visualizations:** Results are displayed using interactive gauges and charts from Plotly.
-   **Personalized Recommendations:** The app provides actionable advice based on the user's input and prediction results.

---

## 🧠 Models Used

The project employs a variety of machine learning techniques to provide a holistic analysis:

-   **Classification Models:**
    -   🌳 **Decision Tree:** For a rule-based, interpretable prediction.
    -   📊 **Naive Bayes:** A probabilistic classifier that works well with high-dimensional data.
    -   🧠 **Neural Network (MLP):** A deep learning model for capturing complex patterns.
-   **Regression Model:**
    -   📈 **Linear Regression:** To predict the numerical `Addicted_Score`.
-   **Clustering Model:**
    -   👥 **K-Means Clustering:** To group similar users based on their behavioral patterns.

---

## 🚀 Live Demo & Visuals

![demo video](streamlit-streamlit_app-2025-12-11-00-11-38.webm)

[App Demo](https://social-media-addiction-vs-relationships.streamlit.app/)

---

## 🛠️ Tech Stack

-   **Python 3.x**
-   **Data Analysis & ML:** Pandas, NumPy, Scikit-learn
-   **Web Framework:** Streamlit
-   **Data Visualization:** Matplotlib, Seaborn, Plotly
-   **Notebook Environment:** Jupyter
-   **Serialization:** Joblib

---

## ⚙️ Setup and Installation

Follow these steps to set up the project on your local machine.

**1. Clone the Repository**
```bash
git clone https://github.com/FortunExLV4/Social_Media_Addiction_vs_Relationships.git
cd Social_Media_Addiction_vs_Relationships
```

**2. Create a Virtual Environment (Recommended)**
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
All required libraries are listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

---

## 🏃‍♀️ How to Run

The project has two main components: the Jupyter Notebook for model training and the Streamlit app for the user interface.

**Step 1: Train the Models (Run the Jupyter Notebook)**

First, you must run the Jupyter Notebook to train the models and generate the necessary `.pkl` files.

1.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2.  Open the `Social_Media_Addiction_Prediction.ipynb` file.
3.  Run all cells from top to bottom. This will create a `models/` directory and populate it with the trained models and transformers.

**Step 2: Run the Streamlit Web Application**

Once the models are saved, you can launch the web app.

1.  Open your terminal or command prompt in the project's root directory.
2.  Run the following command:
    ```bash
    streamlit run streamlit_app.py
    ```
3.  The application will open in your default web browser.

---

## 📂 Directory Structure
```
.
├── 📄 archive/
│   └── Students Social Media Addiction.csv   # Original dataset
├── 📂 models/
│   ├── decision_tree_model.pkl
│   ├── kmeans_model.pkl
│   ├── label_encoders.pkl
│   ├── ... (other saved models)
├── 📂 visualizations/
│   ├── classification_comparison.png
│   ├── ... (other saved plots)
├── 📜 Social_Media_Addiction_Prediction.ipynb # Jupyter Notebook for training
├── 📜 streamlit_app.py                        # Streamlit web application
├── 📜 requirements.txt                        # Project dependencies
└── 📜 README.md                               # This file
```

---

## 📄 Disclaimer

This application provides predictions based on machine learning models and is intended for educational and informational purposes only. The results should not be considered as a substitute for professional medical or psychological advice, diagnosis, or treatment.

```