import streamlit as st
import pickle

# Load the model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Create input fields
st.markdown("<h2 style='text-align: center;'>Institute of Technology of Cambodia</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Department of Applied Mathematics and Statisics</h3>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Mini Project</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Suicide Predictor</h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: right;'>Deploy By Ton Chamnan</h5>", unsafe_allow_html=True)

# Text area for user input
user_input = st.text_area('Enter your text here:')

# Mapping for predictions
prediction_mapping = {0: 'Suicide', 1: 'Non-Suicide'}

# Predict button
if st.button('Predict'):
    user_input_tfidf = vectorizer.transform([user_input])
    prediction = model.predict(user_input_tfidf)[0]
    prediction_label = prediction_mapping[prediction]
    st.write(f'Prediction: {prediction_label}')
