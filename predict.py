import sys
import torch
import torch.nn as nn
from PIL import Image
from transformers import ViTModel, ViTImageProcessor

# === CONFIG ===
MODEL_PATH = 'vit_pair_classifier.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# === MODEL ===
class ViTPairClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.classifier = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, pair_imgs):
        img1, img2 = pair_imgs[:, 0], pair_imgs[:, 1]
        feat1 = self.vit(pixel_values=img1).last_hidden_state[:, 0]
        feat2 = self.vit(pixel_values=img2).last_hidden_state[:, 0]
        combined = torch.cat([feat1, feat2], dim=1)
        return self.classifier(combined)

# === PREDICT FUNCTION ===
def predict(image1_path, image2_path):
    # Load model
    model = ViTPairClassifier().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Load and process images
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    img1 = Image.open(image1_path).convert("RGB")
    img2 = Image.open(image2_path).convert("RGB")
    pixel_values = processor([img1, img2], return_tensors="pt")["pixel_values"].unsqueeze(0).to(DEVICE)


    # Inference
    with torch.no_grad():
        output = model(pixel_values)
        prob = torch.sigmoid(output).item()
        prediction = int(prob > 0.5)

    # Print results
    print(f"\nPrediction result: {prediction}")
    if prediction == 0:
        print("→ Model thinks the ORIGINAL image is FIRST.")
    else:
        print("→ Model thinks the ORIGINAL image is SECOND.")

# === ENTRY POINT ===
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict.py <image1_path> <image2_path>")
        sys.exit(1)

    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    predict(img1_path, img2_path)
