import tensorflow
import streamlit as st
import pickle
from tensorflow.keras.utils import pad_sequences
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('my_model.keras')
with open('additional_info.pkl', 'rb') as file:
    loaded_additional_info = pickle.load(file)

embedding_size = loaded_additional_info['hyperparameters']['embedding_size']
lstm_units = loaded_additional_info['hyperparameters']['lstm_units']
description = loaded_additional_info['description']

tk = pickle.load(open('tokeniser.pkl', 'rb'))

st.title('Text Generator')
st.write('Try typing names of Deep learning scientists')
result = None

text = st.text_input("Enter scientist name")
length = st.text_input("Enter the required length of the sentence")
length = int(length) if length.isdigit() else 20


def word_predictor(text, len):
    for x in range(len):
        sq = tk.texts_to_sequences([text])
        sqe = pad_sequences(sq, maxlen=24 - 1, padding='pre')
        re = np.argmax(model.predict(sqe))
        for word, index in tk.word_index.items():
            if index == re:
                text = text + " " + word
    return text


if st.button("Generate"):
    data = word_predictor(text, length)
    st.write(data)
else:
    st.warning("Generation result is empty. Please make sure to generate first.")

st.write("Note: This machine learning model is trained on less amount of data")
