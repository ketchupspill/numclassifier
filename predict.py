import torch
from models.simple_nn import SimpleNN
from models.model2 import MDL
from PIL import Image
import torchvision.transforms as transforms

# --- 1. Load the trained model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MDL().to(device)
model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
model.eval()  # evaluation mode

# --- 2. Load your image ---
# Replace 'my_digit.png' with your image file
img = Image.open("image.png").convert("L")  # convert to grayscale

# --- 3. Preprocess the image ---
transform = transforms.Compose([
    transforms.Resize((28, 28)),   # ensure 28x28 pixels
    transforms.ToTensor()          # convert to tensor (0-1)
])
img_tensor = transform(img).unsqueeze(0).to(device)  # add batch dimension: [1, 1, 28, 28]

# --- 4. Predict ---
with torch.no_grad():  # no need to calculate gradients
    output = model(img_tensor)              # forward pass
    _, predicted = torch.max(output, 1)    # pick highest score

print(f"Predicted digit: {predicted.item()}")

