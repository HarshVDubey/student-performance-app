import streamlit as st  # Import Streamlit for web app interface
import pandas as pd  # Pandas for data handling
import numpy as np  # NumPy for numerical operations
import pickle  # Pickle to load the trained model
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Preprocessing utilities

# Function to load the pre-trained model, scaler, and label encoder
def load_model():
    with open('Linear_Regression_model.pkl', 'rb') as file:  # Load the model file
        model, scaler, le = pickle.load(file)  # Unpack the saved model, scaler, and encoder
    return model, scaler, le  # Return the loaded objects

# Function to preprocess user input before feeding it into the model
def preprocessing_input_data(data, scaler, le):
    # Convert categorical feature 'Extracurricular Activities' into numerical form
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])  # Convert dictionary into DataFrame
    df_transformed = scaler.transform(df)  # Normalize input data using StandardScaler
    return df_transformed  # Return preprocessed data

# Function to make predictions using the loaded model
def predict_data(data):
    model, scaler, le = load_model()  # Load the model and preprocessing tools
    preprocessed_data = preprocessing_input_data(data, scaler, le)  # Preprocess input data
    prediction = model.predict(preprocessed_data)  # Make prediction
    return round(prediction[0], 2)  # Return predicted score rounded to 2 decimal places

# Main function for Streamlit app
def main():
    st.title("Student Performance Prediction")  # App title
    st.write("Enter your details to get a prediction for your score.")  # Description

    # Collect user input values
    hour_studied = st.number_input("Hours studied", min_value=1, max_value=10, value=5)
    Previous_Scores = st.number_input("Previous Scores", min_value=40, max_value=100, value=70)
    Extracurricular_Activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])
    Sleep_Hours = st.number_input("Sleep Hours", min_value=4, max_value=10, value=7)
    Question_Papers = st.number_input("Question Papers", min_value=0, max_value=10, value=5)

    # Button to trigger prediction
    if st.button("Predict Your Score"):
        # Store user input in a dictionary
        user_data = {
            "Hours Studied": hour_studied,
            "Previous Scores": Previous_Scores,
            "Extracurricular Activities": Extracurricular_Activities,
            "Sleep Hours": Sleep_Hours,
            "Sample Question Papers Practiced": Question_Papers
        }
        
        prediction = predict_data(user_data)  # Get prediction result
        st.success(f"Your predicted score is: {prediction}")  # Display the prediction

# Run the app
if __name__ == "__main__":
    main()
