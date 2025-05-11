import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Mapping class index to original TrashNet labels
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Mapping original labels to general categories
category_map = {
    'cardboard': 'Recyclable',
    'glass': 'Recyclable',
    'metal': 'Recyclable',
    'paper': 'Recyclable',
    'plastic': 'Recyclable',
    'trash': 'Landfill'
}

# Image transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# Load model
@st.cache_resource
def load_model(model_path):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 6),  # 6 classes from TrashNet
        nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def predict_image(model, image):
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image_tensor)
        _, pred = torch.max(output, 1)
    class_label = class_names[pred.item()]
    return class_label, category_map[class_label]


# --- Streamlit UI ---
st.title("♻️ Waste Classifier")
st.write("Upload a waste image to classify it as **Recyclable** or **Landfill**")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model("/src/model_output/trashnet.pt")
    class_label, general_category = predict_image(model, image)

    st.markdown(f"### 🔍 Predicted Class: `{class_label}`")
    st.markdown(f"### ♻️ General Category: `{general_category}`")

    # Optional: Show softmax confidence scores
    if st.checkbox("Show Class Probabilities"):
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.exp(output).squeeze().numpy()

        fig, ax = plt.subplots()
        ax.barh(class_names, probs)
        ax.set_xlabel("Confidence")
        ax.set_title("Class Probabilities")
        st.pyplot(fig)
