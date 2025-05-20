import os
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import ViTImageProcessor
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from model import ViTPairClassifier

# === Config ===
MODEL_PATH = "vit_pair_classifier.pth"
JSON_PATH = "archive/similar/test_subset_styletransfer.json"
ROOT_DIR = "archive"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load model ===
print("[INFO] Loading model...")
model = ViTPairClassifier().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# GradCAM setup (adjust target layer as needed)
target_layers = [model.vit.encoder.layer[-1].output]
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=(DEVICE == "cuda"))

# === Load JSON ===
with open(JSON_PATH, "r") as f:
    pairs = json.load(f)["style_transfer"]

# === Visualization ===
def draw_result(img1, img2, cam_image, predicted_label, actual_label, correct, pred_conf):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    ax[2].imshow(cam_image)

    ax[0].set_title("Left")
    ax[1].set_title("Right")
    ax[2].set_title("GradCAM Heatmap")

    verdict = f"{'✓' if correct else '✗'} Predicted: {'Right' if predicted_label == 1 else 'Left'} is original"
    fig.suptitle(f"{verdict} | Confidence: {pred_conf:.2f}", fontsize=14)

    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.show()

# === Batch inference ===
correct = 0
total = 0
confidences = []

for idx, val in enumerate(pairs.values()):
    orig_path = os.path.join(ROOT_DIR, val["original"])
    gen_path = os.path.join(ROOT_DIR, val["generated"])

    if not os.path.exists(orig_path) or not os.path.exists(gen_path):
        continue

    img_orig = Image.open(orig_path).convert("RGB").resize((224, 224))
    img_gen = Image.open(gen_path).convert("RGB").resize((224, 224))

    rgb_img_orig = np.float32(img_orig) / 255

    # Randomly select the original image's position
    if random.random() < 0.5:
        inputs = processor([img_orig, img_gen], return_tensors="pt")["pixel_values"].unsqueeze(0).to(DEVICE)
        actual_label = 0  # Original on the left
        vis_img1, vis_img2 = img_orig, img_gen
    else:
        inputs = processor([img_gen, img_orig], return_tensors="pt")["pixel_values"].unsqueeze(0).to(DEVICE)
        actual_label = 1  # Original on the right
        vis_img1, vis_img2 = img_gen, img_orig

    with torch.no_grad():
        output = model(inputs)
        pred_score = torch.sigmoid(output).item()
        predicted_label = 1 if pred_score > 0.5 else 0

    is_correct = predicted_label == actual_label
    confidences.append(pred_score)
    total += 1
    if is_correct:
        correct += 1

    # GradCAM visualization
    grayscale_cam = cam(input_tensor=inputs, targets=None)[0]
    cam_image = show_cam_on_image(rgb_img_orig, grayscale_cam, use_rgb=True)

    draw_result(vis_img1, vis_img2, cam_image, predicted_label, actual_label, is_correct, pred_score)

    print(f"[{idx}] Actual: {actual_label}, Predicted: {predicted_label}, Correct: {is_correct}, Confidence: {pred_score:.2f}")

# === Final Stats ===
accuracy = correct / total if total else 0
avg_conf = sum(confidences) / len(confidences) if confidences else 0
print(f"\n[SUMMARY] Accuracy: {accuracy:.2%} | Average confidence: {avg_conf:.2f} ({total} samples)")
