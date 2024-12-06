import numpy as np
import pandas as pd


def load_wisconsin_breast_cancer_data(csv_path, test_size=0.2, random_state=42):
    """
    Load and preprocess the Wisconsin Breast Cancer dataset from the given CSV with headers.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the dataset.
    test_size : float
        Proportion of the dataset to include in the test split.
    random_state : int
        Random seed for shuffling the data.

    Returns
    -------
    X_train : np.ndarray
        Training feature matrix of shape (n_train_samples, n_features).
    X_test : np.ndarray
        Test feature matrix of shape (n_test_samples, n_features).
    y_train : np.ndarray
        Training labels of shape (n_train_samples,).
    y_test : np.ndarray
        Test labels of shape (n_test_samples,).
    feature_names : list
        Names of the feature columns.
    """
    # Load the dataset with headers
    df = pd.read_csv(csv_path)

    # The dataset columns as given:
    # "id", "diagnosis", followed by numerous feature columns.
    # We'll use all columns except id and diagnosis as features.
    # diagnosis is either 'M' or 'B'
    feature_columns = [col for col in df.columns if col not in ["id", "diagnosis"]]

    X = df[feature_columns].values
    y = df["diagnosis"].values

    # Convert diagnosis to binary: M=1, B=0
    y = np.where(y == "M", 1, 0).astype(np.float32)

    # Normalize features: (X - mean) / std
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-9
    X = (X - X_mean) / X_std

    # Shuffle and split into train/test sets
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    num_test = int(len(X) * test_size)
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test, feature_columns


if __name__ == "__main__":
    # Example usage:
    csv_file_path = "data.csv"  # Update path if needed
    X_train, X_test, y_train, y_test, feature_names = load_wisconsin_breast_cancer_data(
        csv_file_path
    )

    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    print("Number of features:", len(feature_names))
    print("Sample training labels:", y_train[:10])
    print("Feature names:", feature_names)
