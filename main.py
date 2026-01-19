import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    #Load the CSV file into a pandas Dataframe

    df = pd.read_csv(filepath)
    return df

def check_missing_values(df):
    #Returns the number of missing values in each column

    missing_values = df.isnull().sum()
    return missing_values

def churn_balance(df):
    # Returns churn distribution statistics

    total = len(df)
    churn_count = df['Exited'].sum()
    non_churn_count = total - churn_count
    churn_rate = churn_count / total
    return{
        "total": total,
        "churned": churn_count,
        "not_churned": non_churn_count,
        "churn_rate": churn_rate
    }

def descriptive_statistics(df):
    # Returns descriptive statistics for numerical columns

    stats = df.describe()
    return stats

def remove_irrelevant_columns(df):
    # Remove columns that are not relevant for analysis(e.g.,RowNumber,CustomerId,Surname,Complain,Satisfacton Score,Card Type,Point Earned)

    df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Complain', 'Satisfaction Score', 'Card Type', 'Point Earned'], axis=1, inplace=True)
    return df

def encode_categorical(df):
    # Encode categorical columns (Geography, Gender) using one-hot encoding

    df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
    return df

def split_data(df, test_size=0.2, random_state=42):
    # Split the dataset into training and testing sets

    X = df.drop('Exited', axis=1)
    y = df['Exited']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    #scale numerical features in X_train and X_test using StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


if __name__ == "__main__":
    #Load the dataset
    df = load_data("dataset/Customer-Churn-Records.csv")

    #Get missing values
    missing_values = check_missing_values(df)
    print("Missing values in each column:")
    print(missing_values)

    print("\n")

    #Get churn distribution
    churn_stats = churn_balance(df)
    print(f"Total customers: {churn_stats['total']}")
    print(f"Churned customers: {churn_stats['churned']}")
    print(f"Non-churned customers: {churn_stats['not_churned']}")
    print(f"Churn rate: {float(churn_stats['churn_rate']):.2%}")

    print("\n")

    #Get descriptive statistics
    stats = descriptive_statistics(df)
    print("Descriptive statistics for numerical features:")
    print(stats)


    print("\n")

    df = remove_irrelevant_columns(df)
    df = encode_categorical(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    print(df.head(5))
    