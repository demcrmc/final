import os
import pandas as pd
from joblib import load
from flask import Flask, request, jsonify
from flask_cors import CORS
from category_encoders import BinaryEncoder

# Load model
model = load('decsiion_tree.joblib')

# Load dataset
x = pd.read_csv('student_depression_dataset.csv')

# Fit encoder
categorical_features = ['Gender', 'Profession', 'Dietary Habits', 
                        'Have you ever had suicidal thoughts ?', 
                        'Family History of Mental Illness', 'Sleep Duration', 'Financial Stress']
encoder = BinaryEncoder()
if all(feature in x.columns for feature in categorical_features):
    encoder.fit(x[categorical_features])

# Initialize app
api = Flask(__name__)
CORS(api)

@api.route("/", methods=["GET", "HEAD"])
def index():
    return "Student Depression Prediction API is running", 200

@api.route('/api/sd_prediction', methods=['POST'])
def prediction_depression():
    data = request.json['inputs']
    input_df = pd.DataFrame(data)

    # Encode categorical features
    input_encoded = encoder.transform(input_df[categorical_features])
    input_df = input_df.drop(categorical_features, axis=1)
    final_input = pd.concat([input_df.reset_index(drop=True), input_encoded.reset_index(drop=True)], axis=1)

    # Get prediction probabilities
    prediction_proba = model.predict_proba(final_input)
    predictions = (prediction_proba[:, 1] > 0.5).astype(int)

    # Build response
    response = [{
        "Depression Probability": round(float(prob[1]) * 100, 2),
        "Prediction": "Likely to have depression" if pred == 1 else "Not likely to have depression"
    } for prob, pred in zip(prediction_proba, predictions)]

    return jsonify({"Prediction": response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    api.run(host="0.0.0.0", port=port)
