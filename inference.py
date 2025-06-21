import torch
from PIL import Image
import pickle
import json
from plant_disease_model import PlantDiseaseModel

def predict_image(image_path, model, transform, device, label_encoder):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
    class_name = label_encoder.inverse_transform([pred.item()])[0]
    return class_name

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("class_names.json") as f:
        class_names = json.load(f)
    model = PlantDiseaseModel(len(class_names))
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    with open("inference_transform.pkl", "rb") as f:
        transform = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    image_path = "test_image.jpg"  # Replace with actual image
    print(predict_image(image_path, model, transform, device, label_encoder))

if __name__ == "__main__":
