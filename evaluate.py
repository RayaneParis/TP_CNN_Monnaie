import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay)

def evaluate(model, loader, idx2label, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()

    print(f"\n{'='*50}")
    print(f" Accuracy globale: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f" Score Kaggle top : 0.90171 (-> Xuban Arrieta)")
    print(f" Ecart au top : {0.90171 - accuracy:+.4f}")
    print(f"{'='*50}\n")

    class_names = [idx2label[i] for i in range(len(idx2label))]
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap='Blues')
    ax.set_title('Matrice de confusion sur la classification de pièces (AlexNet)')
    plt.tight_layout()
    plt.savefig('matrix_confusion.png', dpi=150)
    plt.show()
    print("Matrice de confusion validée et enregistrée : matrix_confusion.png")

    return accuracy, all_labels, all_preds


def prediction_single(model, image_path, idx2label, transform, device):
    from PIL import image

    model.eval()
    img = image.open(image_path).convert('RGD')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probas = torch.softmax(output, dim=1)[0]
        predic = torch.arg(probas).item()

    top5_probas, top5_idx = torch.topk(probas, k=min(5, len(idx2label)))
    print(f"\nPrédiction : {idx2label[predic]}")
    print("Top-5 :")
    for p, i in zip(top5_probas.tolist(), top5_idx.tolist()):
        print(f" {idx2label[i]:<40} {p:.4f}")

    return idx2label[predic], probas



