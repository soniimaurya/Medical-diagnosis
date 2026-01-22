import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, names=column_names)

# Split the dataset into features and target variable
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Create the neural network model
model = Sequential([
    Dense(16, input_shape=(X_train.shape[1],), activation='relu'),  # Input layer
    Dense(8, activation='relu'),  # Hidden layer
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)


import streamlit as st

# Streamlit app
st.title('Diabetes Prediction')

# Input fields for user data
pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0)
glucose = st.number_input('Glucose Level', min_value=0.0, max_value=200.0, value=0.0)
blood_pressure = st.number_input('Blood Pressure', min_value=0.0, max_value=122.0, value=0.0)
skin_thickness = st.number_input('Skin Thickness', min_value=0.0, max_value=99.0, value=0.0)
insulin = st.number_input('Insulin Level', min_value=0.0, max_value=846.0, value=0.0)
bmi = st.number_input('BMI', min_value=0.0, max_value=67.1, value=0.0)
diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.0)
age = st.number_input('Age', min_value=0, max_value=120, value=0)

# Prediction button
if st.button('Predict'):
    # Prepare the input data for prediction
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    input_data = scaler.transform(input_data)
    
    # Perform prediction
    prediction = model.predict(input_data)
    diagnosis = 'Diabetic' if prediction > 0.5 else 'Non-Diabetic'
    
    # Display the result
    st.write(f'The model predicts that the person is: {diagnosis}')