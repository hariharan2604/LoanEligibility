from flask import Flask
import json
import socket
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

@app.route('/',methods=['GET'])
def hello_world():
    train_data = pd.read_csv("loan_data_train.csv")

    # Load the test dataset
    test_data = pd.read_csv("loan_data_test.csv")

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

    # Output predictions
    # output_df = pd.DataFrame({'Loan_ID': test_data.index, 'Loan_Status': y_pred})
    # output_df['Loan_Status'] = output_df['Loan_Status'].map({1: 'Y', 0: 'N'})
    test_data['Loan_Status']= y_pred
    test_data['Loan_Status'] = test_data['Loan_Status'].map({1: 'Y', 0: 'N'})

    # Save the test data with predicted loan status to a CSV file
    test_data.to_csv('loan_data_test_with_predictions.csv', index=False)

    # Note: To print accuracy and classification report, you would need the ground truth labels for the test data,
    # which are usually not available in real-world scenarios where you're predicting on unseen data.
    test_data['Loan_Status'] = test_data['Loan_Status'].map({'Y': 1, 'N': 0})
    y_test=test_data['Loan_Status']
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_rep)
    return json.dumps(classification_rep)

@app.route('/test',methods=['GET'])
def cmd():
    return "hariharan data"

if __name__ == '__main__':
    # Get the host's IP address dynamically
    host_ip = socket.gethostbyname(socket.gethostname())
    app.run(host=host_ip, debug=True)
