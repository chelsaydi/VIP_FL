import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

def zeros_like_state_dict(state_dict, device):
    return {k: torch.zeros_like(v, device=device) for k, v in state_dict.items()}

def move_state_to_device(state_dict, device):
    return {k: v.clone().detach().to(device) for k, v in state_dict.items()}

def scaffold_train_local(model, loader, c_global, c_local, class_weights_tensor, device, epochs=1, lr=0.01):
    model = deepcopy(model).to(device)
    model.train()

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    global_state = move_state_to_device(model.state_dict(), device)

    step_count = 0

    for _ in range(epochs):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad += (c_global[name] - c_local[name])

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            step_count += 1

    local_state = move_state_to_device(model.state_dict(), device)

    new_c_local = {}
    for name in local_state:
        new_c_local[name] = c_local[name] - c_global[name] + (global_state[name] - local_state[name]) / step_count

    delta_c = {}
    for name in new_c_local:
        delta_c[name] = new_c_local[name] - c_local[name]

    return local_state, new_c_local, delta_c

def average_model_states(local_states, client_sizes):
    total_size = sum(client_sizes)
    new_state = {}

    for key in local_states[0].keys():
        new_state[key] = sum(
            (client_sizes[i] / total_size) * local_states[i][key]
            for i in range(len(local_states))
        )

    return new_state

def update_server_control(c_global, delta_cs, beta=0.25):
    num_clients = len(delta_cs)
    new_c_global = {}

    for key in c_global.keys():
        avg_delta = sum(delta_c[key] for delta_c in delta_cs) / num_clients
        new_c_global[key] = c_global[key] + beta * avg_delta

    return new_c_global