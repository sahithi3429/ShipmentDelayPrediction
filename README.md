# **Shipment Delay Prediction**

## **Overview**
This project predicts whether a shipment will be delayed or delivered on time based on historical data. The solution uses machine learning models to make predictions and serves them via a REST API built with Flask.

---

## **Project Structure**
```
ShipmentDelayPrediction/
├── main.py                # Code for data preparation and training
├── app.py                 # Flask API code
├── random_forest_model.pkl # Saved trained model
├── shipment_data.csv      # Original dataset
└── README.md              # Project documentation
```

---

## **Requirements**
The following Python libraries are required:
- Python 3.13.1
- Flask
- pandas
- numpy
- scikit-learn

Install them using:
```bash
pip install -r requirements.txt
```

---

## **Setup Instructions**
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd ShipmentDelayPrediction
   ```

2. **Train the Model**:
   Run the following command to train the model and save it:
   ```bash
   python main.py
   ```

3. **Run the Flask API**:
   Start the API server using:
   ```bash
   python app.py
   ```

4. **Test the API**:
   Use Postman or any HTTP client to send POST requests to:
   ```
   http://127.0.0.1:5000/predict
   ```

---

## **API Usage**
- **Endpoint**: `/predict`
- **Method**: `POST`
- **Headers**: 
  ```
  Content-Type: application/json
  ```
- **Payload Example**:
  ```json
  {
      "features": [1000, 5, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0]
  }
  ```
- **Response Example**:
  ```json
  {
      "prediction": "Delayed"
  }
  ```

---

## **Model Performance**
### Logistic Regression:
- **Accuracy**: 99.6%
- **Precision**: 100%
- **Recall**: 99.46%
- **F1 Score**: 99.73%

### Random Forest:
- **Accuracy**: 99.675%
- **Precision**: 100%
- **Recall**: 99.56%
- **F1 Score**: 99.78%

---

## **Future Improvements**
- Add more features to improve model accuracy.
- Enhance the API by deploying it to a cloud platform (e.g., AWS, GCP, or Heroku).
- Incorporate real-time traffic and weather data for dynamic predictions.

---
