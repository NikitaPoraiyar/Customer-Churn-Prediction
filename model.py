import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

def split_data(df, test_size=0.2, random_state=42):
    # Split the dataset into features (X) and target (y), then into training and test sets

    X = df.drop('Exited', axis=1)
    y = df['Exited']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    # Scale numerical features using StandardScaler and return the scaler for later use

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def build_model():
    # Build and return a Multi-layer Perceptron Classifier model

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),  # Hidden layers
        activation = 'relu',          # ReLU activation function
        solver = 'adam',              # Adam optimizer
        learning_rate_init = 0.001,   # Same learning rate
        max_iter = 100,               # Maximum epochs
        early_stopping = True,        # Enable early stopping
        validation_fraction = 0.2,    # Use 20% of training set for validation
        n_iter_no_change = 10,        # Patience = 10
        alpha = 0.0001,               # L2 regularization term
        random_state=42
    )
    return model

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    # Train the model, evaluate it on the test data, and print metrics

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return model

def save_object(obj, filename):
    # Save a Python object to a file using pickle

    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

if __name__ == "__main__":
    #Load dataset
    df = pd.read_csv("dataset/Customer-Churn-Records.csv")

    #Remove irrelevant columns
    df.drop(
        columns=[
            'RowNumber',
            'CustomerId',
            'Surname',
            'Complain',
            'Satisfaction Score',
            'Card Type',
            'Point Earned'
        ],
        inplace=True
    )

    # one-hot encode categorical columns
    df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

    #Split, scale, and build model
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    model = build_model()

    #train and evaluate
    trained_model = train_and_evaluate(model, X_train_scaled, y_train, X_test_scaled, y_test)

    #Save the trained model and scaler
    save_object(scaler, 'scaler.pkl')
    save_object(trained_model, 'mlp_model.pkl')
    print("Model and scaler saved successfully.")

