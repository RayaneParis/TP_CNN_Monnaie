import torch
from dataset import build_dataloaders
from model import AlexNet, get_pretrain_alexnet
from train import training
from evaluate import evaluate

# Configuration 
if __name__ in "__main__":
    DATA_DIR = './kaggle/train'
    CSV_PATH = './kaggle/train.csv'
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LR = 1e-3
    USE_PRETRAINED = True

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device : {device}")

    # Données
    train_loader, val_loader, num_classes, idx2label = build_dataloaders(
        DATA_DIR, CSV_PATH, batch_size = BATCH_SIZE
    )

    # Modèles
    if USE_PRETRAINED:
        model = get_pretrain_alexnet(num_classes)
        print("AlexNET pré-entrainé")
    else:
        model = AlexNet(num_classes=num_classes)
        print("AlexNET from scratch")

    # Entrainement
    history = training(model, train_loader, val_loader, num_epoch=NUM_EPOCHS, lr=LR, device=device)

    # Evaluation 
    model.load_state_dict(torch.load('best_alexnet.pth'))
    accuracy, predics, labels = evaluate(model, val_loader, idx2label, device)

