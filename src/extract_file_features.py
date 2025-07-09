import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np
import pickle

def extract_features(image_dir="../data/satellite_tiles", output_path="../data/tile_features.pkl"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load pre-trained ResNet18
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Identity()  # Remove final classification layer
    model.eval().to(device)

    features = {}
    for fname in os.listdir(image_dir):
        if fname.endswith(".png"):
            path = os.path.join(image_dir, fname)
            img = Image.open(path).convert("RGB")
            input_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model(input_tensor).squeeze().cpu().numpy()
                features[fname] = embedding

    # Save as pickle file
    with open(output_path, "wb") as f:
        pickle.dump(features, f)

    print(f"✅ Extracted and saved features for {len(features)} tiles → {output_path}")

if __name__ == "__main__":
    extract_features()