import os
import re
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Initialize Flask app
app = Flask(__name__)

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Constants
MAX_LENGTH = 100

# Load model and tokenizer
try:
    model = load_model('amazon_model.h5')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise SystemExit("Cannot continue without model")

try:
    with open('tokenizer (2).pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded successfully")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    raise SystemExit("Cannot continue without tokenizer")


def clean_text(text):
    """Clean and preprocess text"""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower().strip()
    text = ' '.join([lemmatizer.lemmatize(word)
                     for word in text.split()
                     if word not in stop_words])
    return text


def generate_wordcloud(text):
    """Generate word cloud image"""
    from wordcloud import WordCloud
    plt.figure(figsize=(10, 5))
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Save image to bytes
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    return base64.b64encode(img_bytes.read()).decode('utf-8')


@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze review sentiment"""
    text = request.form['text']
    cleaned_text = clean_text(text)

    # Generate word cloud
    wordcloud_img = generate_wordcloud(cleaned_text)

    # Tokenize and pad text
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=MAX_LENGTH)

    # Make prediction
    prediction = model.predict(padded, verbose=0)
    confidence = float(prediction[0][0])
    sentiment = 'positive' if confidence > 0.5 else 'negative'

    # Prepare response
    response = {
        'text': text,
        'cleaned_text': cleaned_text,
        'sentiment': sentiment,
        'confidence': round(confidence * 100, 2) if sentiment == 'positive' else round((1 - confidence) * 100, 2),
        'positive_prob': round(confidence * 100, 2),
        'negative_prob': round((1 - confidence) * 100, 2),
        'wordcloud': wordcloud_img
    }

    return render_template('results.html', **response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)