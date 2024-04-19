import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, json
from flask_cors import CORS  # Import CORS from flask_cors module
import socket

app = Flask(__name__)
CORS(app)  # Enable CORS for your Flask app


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    train_data = pd.read_csv("loan_data_train.csv")

    # Load the test dataset
    test_data = pd.DataFrame(data)

    # Drop irrelevant features (Loan_ID)
    train_data.drop("Loan_ID", axis=1, inplace=True)
    test_data.drop("Loan_ID", axis=1, inplace=True)

    # Handle missing values
    train_data.ffill(inplace=True)  # Forward fill missing values
    test_data.ffill(inplace=True)  # Forward fill missing values

    # Encode categorical variables
    encoder = LabelEncoder()
    categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    for col in categorical_cols:
        train_data[col] = encoder.fit_transform(train_data[col])
        test_data[col] = encoder.transform(test_data[col])  # Use transform instead of fit_transform

    # Manually encode 'Dependents' column to ensure consistency
    dependents_mapping = {'0': 0, '1': 1, '2': 2, '3+': 3}
    train_data['Dependents'] = train_data['Dependents'].map(dependents_mapping)
    test_data['Dependents'] = test_data['Dependents'].map(dependents_mapping)

    # Encode target variable 'Loan_Status' in the training dataset
    train_data['Loan_Status'] = train_data['Loan_Status'].map({'Y': 1, 'N': 0})

    # Split training data into features and target variable
    X_train = train_data.drop("Loan_Status", axis=1)
    y_train = train_data["Loan_Status"]

    # Scale numerical features in the training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Preprocess test data
    # Drop irrelevant features from test data
    X_test = test_data  # Assuming all columns except 'Loan_ID' are relevant

    # Scale numerical features in the test data
    X_test = scaler.transform(X_test)

    # Predict on the test set
    y_pred = model.predict(X_test)
    y_pred_list = y_pred.tolist()

    # Serialize the list to JSON format
    return {"Loan_Status": json.dumps(y_pred_list[0])}


hostip = socket.gethostbyname(socket.gethostname())
app.run(hostip)
