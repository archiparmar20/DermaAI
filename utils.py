import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import TRAIN_DIR, TEST_DIR

# Data transforms
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dataloaders(batch_size=32, num_workers=4):
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=TRAIN_TRANSFORMS)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=TEST_TRANSFORMS)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader, train_dataset.classes

def predict_image(model, image, device):
    from PIL import Image
    transform = TEST_TRANSFORMS
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)
    return pred.item(), confidence.item()
