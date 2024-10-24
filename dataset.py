import torch
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def iris_preprocess():
    """
    Preprocesses the Iris dataset by splitting the data into features and target, scaling the features, and 
    converting the data into PyTorch tensors. Additionally, it creates train and test data loaders for model training.

    :return: A tuple containing:
        - scaler: The fitted StandardScaler used to scale the features.
        - X: The scaled feature matrix as a NumPy array.
        - y: The target labels as a NumPy array.
        - iris_df: A DataFrame representation of the Iris dataset.
        - train_loader: DataLoader object for the training set.
        - test_loader: DataLoader object for the testing set.
    """
    iris = datasets.load_iris()
    
    sepal_length, sepal_width, petal_length, petal_width = zip(*iris['data'])

    iris_dict = {'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width,
            'target': iris['target'] }

    iris_df = pd.DataFrame(iris_dict)

    X = iris_df.drop('target', axis=1).values
    y = iris_df['target'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return scaler, X, y, iris_df, train_loader, test_loader

def pima_preprocess(path):
    """
    Preprocesses the Pima dataset by splitting the data into features and target, scaling the features, and 
    converting the data into PyTorch tensors. Additionally, it creates train and test data loaders for model training.
    
    :param path: Path of Pima Indians Diabetes Dataset.
    :return: A tuple containing:
        - scaler: The fitted StandardScaler used to scale the features.
        - X: The scaled feature matrix as a NumPy array.
        - y: The target labels as a NumPy array.
        - iris_df: A DataFrame representation of the Iris dataset.
        - train_loader: DataLoader object for the training set.
        - test_loader: DataLoader object for the testing set.
    """
    pima_df = pd.read_csv(path)

    X_pima = pima_df.drop('Outcome', axis=1).values
    y_pima = pima_df['Outcome'].values

    scaler = StandardScaler()
    X_pima = scaler.fit_transform(X_pima)

    X_tensor_pima = torch.tensor(X_pima, dtype=torch.float32)
    y_tensor_pima = torch.tensor(y_pima, dtype=torch.long)

    pima_dataset = TensorDataset(X_tensor_pima, y_tensor_pima)

    train_size = int(0.8 * len(pima_dataset))
    test_size = len(pima_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(pima_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    pima_df['target'] = pima_df['Outcome']
    pima_df.drop(['Outcome'], axis = 1, inplace = True)

    return scaler, X_pima, y_pima, pima_df, train_loader, test_loader
