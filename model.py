import os
import json
import random
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor, ViTModel

# === CONFIG ===
JSON_PATH = 'archive/similar/train_subset_styletransfer.json'
ROOT_DIR = 'archive'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8
EPOCHS = 5

# === Dataset ===
class StyleTransferPairDataset(Dataset):
    def __init__(self, json_file, root_dir):
        with open(json_file, 'r') as f:
            data = json.load(f)["style_transfer"]

        self.pairs = []
        for entry in data.values():
            orig_path = os.path.join(root_dir, entry['original'])
            gen_path = os.path.join(root_dir, entry['generated'])
            if os.path.exists(orig_path) and os.path.exists(gen_path):
                self.pairs.append((orig_path, gen_path))

        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        orig_path, gen_path = self.pairs[idx]
        orig_img = Image.open(orig_path).convert("RGB")
        gen_img = Image.open(gen_path).convert("RGB")

        # Randomize position
        if random.random() < 0.5:
            imgs = [orig_img, gen_img]
            label = 0  # original is first
        else:
            imgs = [gen_img, orig_img]
            label = 1  # original is second

        inputs = self.processor(imgs, return_tensors="pt")['pixel_values']
        return inputs, torch.tensor(label, dtype=torch.float32)

# === Model ===
class ViTPairClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
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

# === Train function ===
def train():
    dataset = StyleTransferPairDataset(JSON_PATH, ROOT_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("[INFO] Dataset loaded.")
    model = ViTPairClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"[INFO] Training on {len(dataset)} pairs for {EPOCHS} epochs.")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(loader):
            imgs, labels = imgs.to(DEVICE), labels.unsqueeze(1).to(DEVICE)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[INFO] Epoch {epoch+1}: Loss = {total_loss / len(loader):.4wf}")

    print("[INFO] Saving model to 'vit_pair_classifier.pth'...")
    torch.save(model.state_dict(), "vit_pair_classifier.pth")
    print("[INFO] Model saved successfully.")

if __name__ == "__main__":
    train()
    print("[INFO] Training complete.")