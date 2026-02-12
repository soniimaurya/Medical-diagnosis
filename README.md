🩺 Medical Diagnose – AI-Based Diabetes Prediction System

📌 Project Overview
Medical Diagnose is an AI-powered machine learning application that predicts whether 
a person is diabetic or not based on medical parameters such as glucose level, BMI, age, blood pressure, etc.
The project uses a Neural Network model and a Streamlit web interface to provide real-time predictions in a user-friendly way.

🚀 Features

✅ Diabetes prediction using Machine Learning (Neural Network)
✅ Real-time prediction through Streamlit web app
✅ Data preprocessing and feature scaling
✅ Model training and evaluation
✅ User-friendly UI for medical input
✅ Accuracy and performance evaluation

🧠 Technologies Used

Python
NumPy & Pandas
Scikit-learn
TensorFlow / Keras
Streamlit
Machine Learning & Deep Learning

📊 Dataset
The project uses the PIMA Indians Diabetes Dataset:
Source:
https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
Features:
Pregnancies
Glucose
Blood Pressure
Skin Thickness
Insulin
BMI
Diabetes Pedigree Function
Age
Outcome (0 = Non-Diabetic, 1 = Diabetic)

⚙️ Project Structure
Medical-diagnosis/
│── app.py               # Streamlit web app
│── train_model.py       # Model training script
│── diabetes_model.h5    # Saved ML model
│── scaler.pkl           # Feature scaler
│── requirements.txt     # Required libraries
│── README.md            # Project documentation

🛠️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/soniimaurya/Medical-diagnosis.git
cd Medical-diagnosis

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train the Model (Optional)
python train_model.py

4️⃣ Run the Application
streamlit run app.py


📈 Model Performance

Neural Network-based binary classification
Accuracy evaluated using:
Accuracy Score
Classification Report
Confusion Matrix

💡 Use Cases

Medical decision support
Healthcare analytics
AI-based health prediction systems
Educational ML projects
