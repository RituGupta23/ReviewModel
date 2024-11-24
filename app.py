from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and vectorizer
with open('text_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as tfidf_file:
    tfidf = pickle.load(tfidf_file)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the review from the POST request
    data = request.get_json()
    review = data['review']

    # Vectorize the review using the saved TF-IDF vectorizer
    review_tfidf = tfidf.transform([review])

    # Predict the label using the loaded model
    prediction = model.predict(review_tfidf)
    
    # Map the prediction to the respective label
    label = 'Seller Issue' if prediction[0] == 0 else 'Logistic Issue'

    # Return the prediction as a JSON response
    return jsonify({'label': label})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5002)
