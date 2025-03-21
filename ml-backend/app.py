import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
import joblib
import numpy as np

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

# Configure FLASK_DEBUG from environment variable
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get app root directory
MODEL_DIR = os.path.join(BASE_DIR, 'models')  # Define models directory

# Load models using absolute paths
model = joblib.load(os.path.join(MODEL_DIR, 'bagging_model.joblib'))
lda = joblib.load(os.path.join(MODEL_DIR, 'lda_model.joblib'))
ordinal_enc = joblib.load(os.path.join(MODEL_DIR, 'ordinal_enc.joblib'))
label_encoders = {
    'Gender': joblib.load(os.path.join(MODEL_DIR, 'Gender_label_enc.joblib')),
    'City': joblib.load(os.path.join(MODEL_DIR, 'City_label_enc.joblib')),
    'Membership Type': joblib.load(os.path.join(MODEL_DIR, 'Membership Type_label_enc.joblib'))
}
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))



@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        features = data['features']
        
        features['Gender'] = label_encoders['Gender'].transform([features['Gender']])[0]
        features['City'] = label_encoders['City'].transform([features['City']])[0]
        features['Membership Type'] = label_encoders['Membership Type'].transform([features['Membership Type']])[0]


        numerical_columns = ['Age', 'Total Spend', 'Items Purchased', 'Average Rating', 'Days Since Last Purchase']
        numerical_values = [features[col] for col in numerical_columns]
        numerical_values_scaled = scaler.transform([numerical_values])[0]

        for i, col in enumerate(numerical_columns):
            features[col] = numerical_values_scaled[i]


        processed_features = np.array([
            features['Gender'],
            features['Age'],
            features['City'],
            features['Membership Type'],
            features['Total Spend'],
            features['Items Purchased'],
            features['Average Rating'],
            features['Discount Applied'],  
            features['Days Since Last Purchase']
        ]).reshape(1, -1)

        
        lda_transformed = lda.transform(processed_features)
        prediction = model.predict(lda_transformed)

        decoded_satisfaction_level = ordinal_enc.inverse_transform([[prediction[0]]])[0][0]

        return jsonify({'prediction': decoded_satisfaction_level})
    

    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run()
