import torch
from PIL import Image
import pickle
import json
from torchvision import transforms
from plant_disease_model import PlantDiseaseModel

def predict_image(image_path, model, transform, device, encoder):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out = model(img)
        _, pred = torch.max(out, 1)
    return encoder.inverse_transform([pred.item()])[0]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("class_names.json") as f:
        classes = json.load(f)
    with open("label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("inference_transform.pkl", "rb") as f:
        transform = pickle.load(f)
    model = PlantDiseaseModel(len(classes))
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)

    image_path = "test_leaf.jpg"  # Replace with your image path
    result = predict_image(image_path, model, transform, device, encoder)
    print(f"Predicted class: {result}")

if __name__ == "__main__":
    main()
