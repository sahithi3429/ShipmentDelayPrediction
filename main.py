import pandas as pd

# Load the dataset
df = pd.read_csv('shipment_data.csv')

# Display the first few rows of the dataset
print(df.head())
print(df.info())
print(df.describe())

# Handle missing values in 'Vehicle Type'
df['Vehicle Type'].fillna(df['Vehicle Type'].mode()[0], inplace=True)

# Convert date columns to datetime
df['Shipment Date'] = pd.to_datetime(df['Shipment Date'])
df['Planned Delivery Date'] = pd.to_datetime(df['Planned Delivery Date'])
df['Actual Delivery Date'] = pd.to_datetime(df['Actual Delivery Date'])

# Drop irrelevant columns
df.drop(['Shipment ID'], axis=1, inplace=True)

# Check the cleaned data
print(df.info())
print(df.head())

# Calculate delivery duration
df['Delivery Duration'] = (df['Actual Delivery Date'] - df['Shipment Date']).dt.days

df.drop(['Shipment Date', 'Planned Delivery Date', 'Actual Delivery Date'], axis=1, inplace=True)

df['Delayed'] = df['Delayed'].apply(lambda x: 1 if x == 'Yes' else 0)

df = pd.get_dummies(df, columns=['Origin', 'Destination', 'Vehicle Type', 'Weather Conditions', 'Traffic Conditions'], drop_first=True)

print(df.head())
print(df.info())

# Separate features and target
X = df.drop(['Delayed'], axis=1)
y = df['Delayed']

from sklearn.model_selection import train_test_split

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the splits
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize and train the logistic regression model
model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_lr.fit(X_train, y_train)

# Make predictions
y_pred = model_lr.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Logistic Regression Metrics:\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")

from sklearn.ensemble import RandomForestClassifier

# Initialize and train the Random Forest model
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Make predictions
y_pred_rf = model_rf.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print(f"\nRandom Forest Metrics:\nAccuracy: {accuracy_rf}\nPrecision: {precision_rf}\nRecall: {recall_rf}\nF1 Score: {f1_rf}")

import pickle

# Save the trained Random Forest model
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(model_rf, file)
print("Model saved successfully!")
