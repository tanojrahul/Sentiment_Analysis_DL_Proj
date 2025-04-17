from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import re

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
try:
    model = load_model('sentiment_analysis_model.h5')
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    model_loaded = True
except (FileNotFoundError, IOError):
    print("Warning: Model or vectorizer file not found. Using dummy sentiment.")
    model = None
    vectorizer = None
    model_loaded = False


# Preprocessing function
def preprocess_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub(r'\d+', '', tweet)
    tweet = tweet.strip()
    return tweet


@app.route('/predict', methods=['POST'])
def predict_sentiment():
    try:
        data = request.get_json()
        tweet = data.get("tweet", "")

        if not tweet:
            return jsonify({"error": "Tweet is required"}), 400

        if not model_loaded:
            # Return dummy sentiment when model is not available
            return jsonify({"sentiment": 0, "note": "Using dummy sentiment (model not loaded)"}), 200

        cleaned_tweet = preprocess_tweet(tweet)
        transformed = vectorizer.transform([cleaned_tweet])
        prediction = model.predict(transformed)

        # Assuming binary classification with sigmoid output
        sentiment = int(prediction[0][0] > 0.5)

        return jsonify({"sentiment": sentiment})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8001, debug=True)
