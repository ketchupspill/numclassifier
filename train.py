import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from models.simple_nn import SimpleNN
from models.model2 import MDL

# --- Hyperparameters ---
batch_size = 64
lr = 0.001
epochs = 5

# --- Data ---
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# --- Model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MDL().to(device)

# --- Loss & optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# --- Training loop ---
for epoch in range(1, epochs+1):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {running_loss/len(train_loader):.4f}")

# --- Save model ---
torch.save(model.state_dict(), "mnist_model.pth")
print("Model saved as mnist_model.pth")
