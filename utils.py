from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import torch 

def save_model(model, path):
    """Save the trained model to the specified path."""
    torch.save(model.state_dict(), path)

def print_classification_report(all_labels, all_preds):
    """Print classification report and confusion matrix."""
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))
    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)