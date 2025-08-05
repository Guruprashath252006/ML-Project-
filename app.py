import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import gradio as gr

# Define the model class
class PlasticNet(nn.Module):
    def __init__(self):
        super(PlasticNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 6)  # Output for 6 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Class labels and extra info
class_names = ["PET", "HDPE (PE)", "PP", "PS", "PC", "Others"]
recycling_methods = {
    "PET": "Mechanical",
    "HDPE (PE)": "Mechanical",
    "PP": "Mechanical",
    "PS": "Chemical",
    "PC": "Mechanical/Chemical",
    "Others": "Specialized/Downcycling"
}
recycled_forms = {
    "PET": "Polyester fabric, packaging",
    "HDPE (PE)": "Containers, piping, bins",
    "PP": "Automotive parts, fiber",
    "PS": "Styrene oil, insulation",
    "PC": "Eyewear frames, layered sheets",
    "Others": "Composite material, energy recovery"
}

# Load model
model = PlasticNet()
model.load_state_dict(torch.load("plasticnet_model.pth", map_location=torch.device("cpu")))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Classification function
def classify_image(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]
        return label, recycling_methods[label], recycled_forms[label]

# Gradio interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(label="Plastic Type"),
        gr.Label(label="♻️ Recycling Method"),
        gr.Label(label="📦 Common Recycled Products")
    ],
    title="Plastic Type Classifier",
    description="Upload an image of plastic to identify its type and view recycling details."
)

iface.launch()
