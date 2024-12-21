import pickle
import numpy as np
from flask import Flask, request, jsonify

# Load the saved model
with open('random_forest_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()
    features = np.array([data['features']])

    # Make prediction
    prediction = loaded_model.predict(features)[0]
    result = "Delayed" if prediction == 1 else "On Time"

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
