
import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# Define your model (PlasticNet)
class PlasticNet(nn.Module):
    def __init__(self):
        super(PlasticNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Load trained model (You need to place model.pth in the same directory)
model = PlasticNet()
model.load_state_dict(torch.load("plasticnet_model.pth", map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

plastic_types = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS", "Other"]

recycling_methods = {
    "PET": "Curbside recycling or deposit programs",
    "HDPE": "Curbside recycling",
    "PVC": "Specialized collection points",
    "LDPE": "Store drop-off recycling",
    "PP": "Curbside recycling",
    "PS": "Avoided or special programs",
    "Other": "Generally not recyclable"
}

recycled_products = {
    "PET": "Clothing, carpets, containers",
    "HDPE": "Pipes, plastic lumber",
    "PVC": "Pipes, tiles",
    "LDPE": "Trash bags, floor tiles",
    "PP": "Batteries, signal lights",
    "PS": "Insulation, rulers",
    "Other": "Mixed uses"
}

def predict_plastic(img):
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        index = int(round(output.item() * (len(plastic_types) - 1)))
        label = plastic_types[index]
        return {
            "Plastic Type": label,
            "♻️ Recycling Method": recycling_methods[label],
            "📦 Common Recycled Products": recycled_products[label]
        }

demo = gr.Interface(fn=predict_plastic, 
                    inputs=gr.Image(type="pil"), 
                    outputs=["text", "text", "text"],
                    title="Plastic Type Identifier + Recycler",
                    description="Upload an image of plastic to identify its type and recycling information")

demo.launch()
