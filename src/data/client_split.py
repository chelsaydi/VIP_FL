import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def create_client_datasets(df_all, feature_cols):
    """
    Create a raw client-level dictionary:
    client_datasets[city] = (X_city, y_city)
    """
    client_datasets = {}

    for city in df_all["city"].unique():
        df_city = df_all[df_all["city"] == city]

        X_city = df_city[feature_cols].values
        y_city = df_city["label_encoded"].values

        client_datasets[city] = (X_city, y_city)

    return client_datasets


def create_client_samples(df_all, feature_cols, max_per_client=100000):
    """
    Create capped client samples for training.
    """
    client_samples = {}

    for city in df_all["city"].unique():
        df_city = df_all[df_all["city"] == city]

        if len(df_city) > max_per_client:
            df_city = df_city.sample(n=max_per_client, random_state=42)

        X_city = df_city[feature_cols].values.astype(np.float32)
        y_city = df_city["label_encoded"].values.astype(np.int64)

        client_samples[city] = (X_city, y_city)

    return client_samples


def print_client_samples_summary(client_samples):
    for city in client_samples:
        print(city, client_samples[city][0].shape, np.unique(client_samples[city][1]))


def create_client_loaders(client_samples, batch_size=256):
    """
    Convert client samples into PyTorch DataLoaders.
    """
    client_loaders = {}

    for city, (X_city, y_city) in client_samples.items():
        X_tensor = torch.tensor(X_city, dtype=torch.float32)
        y_tensor = torch.tensor(y_city, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        client_loaders[city] = loader

    return client_loaders


def print_client_loaders_summary(client_loaders):
    for city in client_loaders:
        print(city, len(client_loaders[city]))


def create_test_set(df_all, feature_cols, n_samples=40000):
    """
    Create a random test sample for FL evaluation.
    """
    df_test = df_all.sample(n=n_samples, random_state=42)

    X_test = torch.tensor(
        df_test[feature_cols].values.astype(np.float32),
        dtype=torch.float32
    )
    y_test = df_test["label_encoded"].values

    return X_test, y_test