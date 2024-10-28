import torch
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

def iris_preprocess():
    """
    Preprocesses the Iris dataset by splitting the data into features and target, scaling the features, and 
    converting the data into PyTorch tensors. Additionally, it creates train and test data loaders for model training.

    :return: A tuple containing:
        - scaler: The fitted StandardScaler used to scale the features.
        - X: The scaled feature matrix as a NumPy array.
        - y: The target labels as a NumPy array.
        - new_df: A DataFrame representation of the Iris dataset.
        - train_loader: DataLoader object for the training set.
        - test_loader: DataLoader object for the testing set.
    """
    iris = datasets.load_iris()

    iris_df = pd.DataFrame(data = iris['data'], columns = iris['feature_names'])
    iris_df['target'] = iris['target']

    X = iris_df.drop('target', axis=1).values
    y = iris_df['target'].values

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    new_df = pd.DataFrame(X, columns = iris_df.columns[:-1])
    new_df['target'] = iris_df['target']

    return scaler, X, y, new_df, train_loader, test_loader

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

    scaler = MinMaxScaler()
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

    new_df = pd.DataFrame(X_pima, columns = pima_df.columns[:-1])
    new_df['target'] = pima_df['target']

    return scaler, X_pima, y_pima, new_df, train_loader, test_loader

def breast_preprocess():
    """
    Preprocesses the Breast Cancer dataset by splitting the data into features and target, scaling the features, and 
    converting the data into PyTorch tensors. Additionally, it creates train and test data loaders for model training.

    :return: A tuple containing:
        - scaler: The fitted StandardScaler used to scale the features.
        - X: The scaled feature matrix as a NumPy array.
        - y: The target labels as a NumPy array.
        - new_breast_df: A DataFrame representation of the Breast Cancer dataset.
        - train_loader: DataLoader object for the training set.
        - test_loader: DataLoader object for the testing set.
    """
    breast_sk = datasets.load_breast_cancer()

    breast_df = pd.DataFrame(data = breast_sk['data'], columns = breast_sk['feature_names'])
    breast_df['target'] = breast_sk['target']

    X = breast_df.drop('target', axis=1).values
    y = breast_df['target'].values

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    new_breast_df = pd.DataFrame(X, columns = breast_df.columns[:-1])
    new_breast_df['target'] = breast_df['target']

    return scaler, X, y, new_breast_df, train_loader, test_loader

def adult_preprocess(path):
    """
    Preprocesses the Adult Income dataset by performing several data cleaning and transformation steps, including
    handling missing values, encoding categorical variables, scaling numerical features, and preparing the data for 
    model training in PyTorch.

    :param path: Path to the Adult Income dataset CSV file.
    :return: A tuple containing:
        - scaler: The fitted MinMaxScaler used to scale the numerical features.
        - X_adult: The scaled feature matrix as a NumPy array.
        - y_adult: The target labels as a NumPy array.
        - adult_df: A DataFrame representation of the preprocessed Adult dataset, including all features.
        - train_loader: DataLoader object for the training set.
        - test_loader: DataLoader object for the testing set.
        - encoded_columns: A list of the names of the columns of the encoded dataset.
    """
    adult_df = pd.read_csv(path)

    adult_df.replace('?', '', inplace = True)
    adult_df = adult_df.replace('', pd.NA).dropna()

    adult_df['target'] = adult_df['income'].apply(lambda x: 0 if x == '<=50K' else 1)
    adult_df = adult_df.drop('income', axis = 1)

    numeric_cols = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    scaler = MinMaxScaler()
    adult_df[numeric_cols] = scaler.fit_transform(adult_df[numeric_cols])

    encoded_adult = pd.get_dummies(adult_df)
    encoded_adult = encoded_adult.astype(float)

    X_adult = encoded_adult.drop('target', axis = 1).values
    y_adult = encoded_adult['target'].values

    X_tensor = torch.tensor(X_adult, dtype=torch.float32)
    y_tensor = torch.tensor(y_adult, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return scaler, X_adult, y_adult, adult_df, train_loader, test_loader, encoded_adult.drop('target', axis = 1).columns
