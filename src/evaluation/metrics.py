import torch
from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(model, X_test, y_test, device):
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(X_test.to(device))
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)
    print(classification_report(y_test, preds, zero_division=0))
    return acc, preds