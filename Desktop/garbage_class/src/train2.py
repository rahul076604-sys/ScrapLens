import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import timm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 10


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


base_dataset = datasets.ImageFolder(root="/content/new_dataset/garbage_classification/")



class_names = base_dataset.classes
NUM_CLASSES = len(class_names) 
print(f"Detected {NUM_CLASSES} classes: {class_names}")




dataset_size = len(base_dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(42)).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:]


train_dataset = Subset(datasets.ImageFolder(root="/content/new_dataset/garbage_classification/", transform=train_transform), train_indices)
val_dataset = Subset(datasets.ImageFolder(root="/content/new_dataset/garbage_classification/", transform=val_transform), val_indices)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.05)

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nFinal Validation Accuracy across {NUM_CLASSES} classes: {(100 * correct / total):.2f}%")

torch.save(model.state_dict(), 'vit_13_waste_classifier.pth')
print("Model saved as vit_13_waste_classifier.pth")
