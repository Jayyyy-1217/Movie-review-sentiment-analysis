import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
word_index=imdb.get_word_index()
reversed={value:key for key,value in word_index.items()}
model=load_model('simplernn.h5')

def decode(encode_review):
    return ' '.join([reversed.get(i-3,'?') for i in encode_review])

def preprocess(text):
    words=text.lower().split()
    encode_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encode_review],maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_text=preprocess(review)
    prediction=model.predict(preprocessed_text)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment,prediction[0][0]

import streamlit as st
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a moview review to classify it as ppsitive or negative')
input=st.text_area('Movie Review')
if st.button('Classify'):
    sentiment,score=predict_sentiment(input)
    st.write(f'Senitment:{sentiment}')
    st.write(f"Prediction score:{score}")
else:
    st.write("Please enter a movie review")
    
    