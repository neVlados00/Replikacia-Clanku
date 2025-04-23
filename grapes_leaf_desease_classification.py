import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Define paths to datasets
train_path = 'PlantVillage/train'
val_path = 'PlantVillage/val'

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load training and validation datasets
train_dataset = ImageFolder(root=train_path, transform=transform)
val_dataset = ImageFolder(root=val_path, transform=transform)

# Filter for grape-related classes (containing "Grape" in the name)
all_classes = train_dataset.classes
grape_class_indices = [i for i, cls in enumerate(all_classes) if 'Grape' in cls]
grape_classes = [all_classes[i] for i in grape_class_indices]
num_grape_classes = len(grape_classes)

# Get indices of grape samples for training dataset
train_grape_indices = [i for i, (_, label) in enumerate(train_dataset.samples) if label in grape_class_indices]

# Limit to 450 images for the training dataset
if len(train_grape_indices) > 450:
    train_grape_indices = np.random.choice(train_grape_indices, 450, replace=False)

# Split training indices into training (80%) and testing (10%) sets
labels = [train_dataset.targets[i] for i in train_grape_indices]
train_idx, test_idx = train_test_split(
    np.arange(len(train_grape_indices)), test_size=0.1111, stratify=labels, random_state=42
)

# Get indices of grape samples for validation dataset
val_grape_indices = [i for i, (_, label) in enumerate(val_dataset.samples) if label in grape_class_indices]

# Create subsets
train_subset = Subset(train_dataset, [train_grape_indices[i] for i in train_idx])
test_subset = Subset(train_dataset, [train_grape_indices[i] for i in test_idx])
val_subset = Subset(val_dataset, val_grape_indices)

# Remap labels to range from 0 to num_grape_classes - 1
label_mapping = {original_label: new_label for new_label, original_label in enumerate(grape_class_indices)}

# Custom dataset class to handle label remapping
class RemappedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, label_mapping):
        self.subset = subset
        self.label_mapping = label_mapping
        self.dataset = subset.dataset

    def __getitem__(self, index):
        x, y = self.subset[index]
        return x, self.label_mapping[y]

    def __len__(self):
        return len(self.subset)

# Create remapped datasets
train_dataset_remapped = RemappedDataset(train_subset, label_mapping)
val_dataset_remapped = RemappedDataset(val_subset, label_mapping)
test_dataset_remapped = RemappedDataset(test_subset, label_mapping)

# Create data loaders with batch size 30
batch_size = 30
train_loader = DataLoader(train_dataset_remapped, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset_remapped, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset_remapped, batch_size=batch_size, shuffle=False)

# Load VGG16 with pre-trained weights
model = models.vgg16(weights='IMAGENET1K_V1')

# Freeze feature layers
for param in model.features.parameters():
    param.requires_grad = False

# Modify classifier
model.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(True),
    nn.Dropout(0.25),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(0.25),
    nn.Linear(4096, num_grape_classes)
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.00001)

# Training function
def train_model(model, train_loader, val_loader, epochs=40):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc / 100)  # Convert to fraction for plotting
        
        val_loss, val_acc = evaluate_model(model, val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc / 100)  # Convert to fraction for plotting
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, Val Accuracy: {val_acc:.2f}%")
    
    # Create a single figure with two subplots for accuracy and loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot accuracy on the left subplot
    ax1.plot(range(1, epochs + 1), train_accuracies, color='blue', label='training accuracy')
    ax1.plot(range(1, epochs + 1), val_accuracies, color='red', label='validations accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Grapes Model: Training and Validation Accuracy')
    ax1.legend()
    
    # Plot loss on the right subplot
    ax2.plot(range(1, epochs + 1), train_losses, color='blue', label='training loss')
    ax2.plot(range(1, epochs + 1), val_losses, color='red', label='validations loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Grapes Model: Training and Validation Loss')
    ax2.legend()
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig('grapes_metrics1.png')
    plt.clf()

# Evaluation function
def evaluate_model(model, loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss = running_loss / len(loader.dataset)
    accuracy = 100 * correct / total
    return loss, accuracy

# Train the model
train_model(model, train_loader, val_loader)

# Evaluate on test set
test_loss, test_accuracy = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Confusion matrix for test set
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Create confusion matrix with numeric labels
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=range(num_grape_classes), yticklabels=range(num_grape_classes))
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion matrix')
plt.savefig('grapes_confusion_matrix1.png')
plt.clf()