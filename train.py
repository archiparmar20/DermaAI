import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import NUM_CLASSES, MODEL_PATH
from utils import get_dataloaders

def train_model(epochs=20, batch_size=32, lr=0.001, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    train_loader, test_loader, classes = get_dataloaders(batch_size, num_workers=2)
    
    # Model
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    
    for epoch in range(epochs):
        # Train
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} Train'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Test
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Test'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_loss_avg = test_loss / len(test_loader)
        test_acc = 100. * correct / total
        test_losses.append(test_loss_avg)
        test_accs.append(test_acc)
        
        scheduler.step()
        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | Test Loss={test_loss_avg:.4f}, Acc={test_acc:.2f}%')
    
    # Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Train Loss')
    plt.plot(test_losses, 'r-', label='Test Loss')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'b-', label='Train Acc')
    plt.plot(test_accs, 'r-', label='Test Acc')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig('training_history.png')
    plt.show()
    
    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')
    return model

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = train_model(epochs=20, batch_size=64, lr=0.001, device=device)
