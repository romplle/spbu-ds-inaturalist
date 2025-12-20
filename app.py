import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# Классы
classes = ['Limenitis arthemis', 'Vanessa cardui', 'Aglais io', 'Vanessa atalanta', 'Nymphalis antiopa']

# Загрузка модели
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)
model.load_state_dict(torch.load('butterfly_classifier.pth'))
model.eval()

# Трансформация для инференса
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def classify_image(image):
    img = Image.fromarray(image)
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]

# Интерфейс Gradio
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(),
    outputs="text",
    title="Классификатор бабочек",
    description="Загрузи изображение бабочки, и модель определит вид."
)

iface.launch()