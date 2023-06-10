import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load the trained Random Forest model
rf_model = joblib.load('rf_model.joblib')

# Define the selected columns
selected_columns = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'College_GPA']

# Create a function to get user input and make predictions
def predict_admit_chance(input_values):
    input_df = pd.DataFrame([input_values], columns=selected_columns)
    prediction = rf_model.predict(input_df)[0]
    return prediction

# Create the Streamlit web app
def main():
    # Set the page title
    st.set_page_config(page_title='Admit Chance Predictor')

    # Set the app header
    st.title('Admit Chance Predictor')

    # Create input fields for user values
    input_values = {}
    input_values['GRE_Score'] = st.number_input('Your GRE score')
    input_values['TOEFL_Score'] = st.number_input('Your TOEFL score')
    input_values['University_Rating'] = st.number_input('Ranking of your university choice')
    input_values['College_GPA'] = st.number_input('College GPA (out of 10)')

    # Create a button for prediction
    if st.button('Predict'):
        # Make predictions and display the result
        prediction = predict_admit_chance(list(input_values.values()))
        prediction_percentage = round(prediction * 100, 2)
        st.success(f'Your admit chance is: {prediction_percentage}%')

# Run the app
if __name__ == '__main__':
    main()
