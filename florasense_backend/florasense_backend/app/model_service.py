"""
Model service for FloraSense AI.

Loads the EfficientNetV2-S checkpoint trained in the Week 1 notebook, and exposes
a single `predict()` method that returns a calibrated top-5 prediction plus a
Grad-CAM visualization (as a base64 PNG), ready to hand straight to the API layer.
"""
import base64
import io
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib

matplotlib.use("Agg")  # headless backend, no display server on a server box
import matplotlib.pyplot as plt

MODEL_DIR = os.getenv("FLORASENSE_MODEL_DIR", "./models")
MODEL_PATH = os.path.join(MODEL_DIR, "florasense_best.pt")
LABELS_PATH = os.path.join(MODEL_DIR, "idx_to_name.json")
CALIBRATION_PATH = os.path.join(MODEL_DIR, "calibration.json")

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.5  # below this, the app should say "not sure" rather than guess

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_eval_tfms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


class FloraSenseModel:
    """Wraps the trained classifier + calibration + Grad-CAM into one object."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.idx_to_name: Dict[int, str] = {}
        self.temperature: float = 1.0
        self.model = None
        self._target_layers = None
        self.loaded = False

    def load(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model weights not found at {MODEL_PATH}. "
                f"Copy florasense_best.pt, idx_to_name.json, calibration.json "
                f"from your Colab export into {MODEL_DIR}/"
            )

        with open(LABELS_PATH) as f:
            raw = json.load(f)
        self.idx_to_name = {int(k): v for k, v in raw.items()}

        with open(CALIBRATION_PATH) as f:
            self.temperature = json.load(f)["temperature"]

        num_classes = len(self.idx_to_name)
        model = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=num_classes)
        state_dict = torch.load(MODEL_PATH, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self.model = model
        self._target_layers = [model.conv_head]  # same layer used in the training notebook
        self.loaded = True

    def _denormalize(self, img_tensor: torch.Tensor) -> np.ndarray:
        mean = np.array(IMAGENET_MEAN)
        std = np.array(IMAGENET_STD)
        img = img_tensor.permute(1, 2, 0).cpu().numpy()
        img = (img * std) + mean
        return np.clip(img, 0, 1)

    def _make_gradcam_image(self, image_tensor: torch.Tensor) -> str:
        """Runs Grad-CAM and returns a base64-encoded PNG (original | heatmap)."""
        input_tensor = image_tensor.unsqueeze(0).to(self.device)

        with GradCAM(model=self.model, target_layers=self._target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor)[0]

        rgb_img = self._denormalize(image_tensor)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(rgb_img)
        axes[0].set_title("Original", fontsize=9)
        axes[0].axis("off")
        axes[1].imshow(visualization)
        axes[1].set_title("Grad-CAM", fontsize=9)
        axes[1].axis("off")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def predict(self, pil_image: Image.Image) -> Tuple[str, float, bool, List[Tuple[str, float]], str]:
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        pil_image = pil_image.convert("RGB")
        image_tensor = _eval_tfms(pil_image)
        input_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            calibrated_probs = F.softmax(logits / self.temperature, dim=1)[0]

        top5_probs, top5_idx = torch.topk(calibrated_probs, k=min(5, len(self.idx_to_name)))
        top5 = [
            (self.idx_to_name.get(idx.item(), f"class_{idx.item()}"), prob.item())
            for idx, prob in zip(top5_idx, top5_probs)
        ]

        pred_species, pred_conf = top5[0]
        is_confident = pred_conf >= CONFIDENCE_THRESHOLD

        gradcam_b64 = self._make_gradcam_image(image_tensor)

        return pred_species, pred_conf, is_confident, top5, gradcam_b64


# Singleton instance imported by main.py
flora_model = FloraSenseModel()
