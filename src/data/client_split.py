import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def create_client_datasets(X_windows, y_windows, meta_df):
    """
    Create client-level datasets from windowed data:
    client_datasets[city] = (X_city, y_city)
    """
    client_datasets = {}

    for city in meta_df["city"].unique():
        city_mask = meta_df["city"] == city

        X_city = X_windows[city_mask]
        y_city = y_windows[city_mask]

        client_datasets[city] = (X_city, y_city)

    return client_datasets


def create_client_samples(X_windows, y_windows, meta_df, max_per_client=None):
    """
    Create standardized client samples from windowed data.

    Parameters
    ----------
    X_windows : np.ndarray
        Windowed feature matrix
    y_windows : np.ndarray
        Window labels
    meta_df : pd.DataFrame
        Metadata for each window, must contain 'city'
    max_per_client : int or None
        If provided, cap each client to at most this many windows
        using deterministic sampling
    """
    client_samples = {}

    for city in meta_df["city"].unique():
        city_mask = meta_df["city"] == city

        X_city = X_windows[city_mask]
        y_city = y_windows[city_mask]

        if max_per_client is not None and len(X_city) > max_per_client:
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(len(X_city), size=max_per_client, replace=False)
            idx = np.sort(idx)

            X_city = X_city[idx]
            y_city = y_city[idx]

        X_city = X_city.astype(np.float32)
        y_city = y_city.astype(np.int64)

        client_samples[city] = (X_city, y_city)

    return client_samples


def print_client_samples_summary(client_samples):
    for city, (X_city, y_city) in client_samples.items():
        print(city, X_city.shape, np.unique(y_city))


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


def create_test_set(X_windows, y_windows, test_size=40000):
    """
    Create a deterministic test set from windowed data.

    Parameters
    ----------
    X_windows : np.ndarray
        Windowed feature matrix
    y_windows : np.ndarray
        Window labels
    test_size : int
        Number of test windows to sample

    Returns
    -------
    X_test : torch.Tensor
    y_test : np.ndarray
    """
    if test_size > len(X_windows):
        test_size = len(X_windows)

    rng = np.random.default_rng(seed=42)
    idx = rng.choice(len(X_windows), size=test_size, replace=False)
    idx = np.sort(idx)

    X_test = torch.tensor(X_windows[idx].astype(np.float32), dtype=torch.float32)
    y_test = y_windows[idx].astype(np.int64)

    return X_test, y_test
