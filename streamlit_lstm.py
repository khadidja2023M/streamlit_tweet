

import os
import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.models import load_model
from keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer



# Create 5 equal-width columns
col1, col2, col3, col4, col5 = st.columns(5)

# Place the buttons inside the columns without using custom CSS for positioning
with col1:
    if st.button("**About us**", key="myycustom"):
        st.write("Disaster Tweet Classification App.")

        
with col3:
    if st.button("**Satisfaction**", key="custom"):
        st.selectbox("Rate your satisfaction (1-5)", range(1, 6))


with col5:
    if st.button("**Contact us**", key="info"):
        st.write("khadidja_mek@hotmail.fr")



st.title('Disaster Tweet Classification 💎')
st.write('This application allows you to classify if its a disaster tweet or not.📈')



st.sidebar.header('Help Menu')
with st.sidebar:
  
  button_clicked = st.button("**Help Menu**")
  if button_clicked:
    st.write("Welcome to my Streamlit app!")
    
    st.write("Enter your text.")

    st.write("Enjoy using the app!")


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Text preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r"http\S|www\S|https\S", '', text, flags=re.MULTILINE)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words)
    return text

train['text'] = train['text'].apply(preprocess_text)
test['text'] = test['text'].apply(preprocess_text)

X_train = train['text']
y_train = train['target']
X_test = test['text']

# CountVectorizer to convert the text data into a matrix of token counts
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
# Split train data into train and validation set
X_train_vectorized, X_val_vectorized, y_train, y_val = train_test_split(X_train_vectorized, y_train, test_size=0.2, random_state=42)



# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Text preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r"http\S|www\S|https\S", '', text, flags=re.MULTILINE)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(stemmer.stem(word) for word in text.split() if word not in stop_words)
    return text

train['text'] = train['text'].apply(preprocess_text)
test['text'] = test['text'].apply(preprocess_text)

X = train['text']
y = train['target']

# Tokenize text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=100)

# Split train data into train and validation set
X_train, X_val, y_train, y_val = train_test_split(X_pad, y, test_size=0.2, random_state=42)

# Load the pre-trained model
model_path = 'LSTM_model.h5'

if os.path.exists(model_path):
    model = load_model(model_path)
    st.write('LSTM model loaded')
else:
    # Build and train the LSTM model if the model file doesn't exist
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=32, input_length=100))
    model.add(LSTM(64, dropout=0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5)
    model.save('LSTM_model.h5')



# User input
user_input = st.text_input("Please enter your text:")

if user_input:
    # Preprocess the input sentence
    sentence = preprocess_text(user_input)
    sentence_seq = tokenizer.texts_to_sequences([sentence])
    sentence_pad = pad_sequences(sentence_seq, maxlen=100)

    # Predict using the loaded LSTM model
    prediction = model.predict(sentence_pad)[0][0]
    result = "Disaster tweet" if prediction > 0.5 else "No disaster"
    st.write("Prediction:", result)
     
    _, accuracy = model.evaluate(X_val, y_val)
    st.write(f"LSTM Model Accuracy: {accuracy:.2f}")
    st.progress(accuracy)

    st.subheader('Author👑')
    st.write('**Khadidja Mekiri**' )