import torch
import torch.nn as nn
import torch.optim as optim

def train_local_fedavg_matched(model, loader, class_weights_tensor, device, epochs=1, lr=0.001):
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.SGD(model.parameters(), lr=lr)

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