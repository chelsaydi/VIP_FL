import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

def train_local(model, loader, device, epochs=1, lr=0.001):
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    return model.state_dict()

def fedavg_aggregate(local_states, client_sizes):
    new_state = deepcopy(local_states[0])
    total_size = sum(client_sizes)

    for key in new_state.keys():
        new_state[key] = sum(
            (client_sizes[i] / total_size) * local_states[i][key]
            for i in range(len(local_states))
        )

    return new_state