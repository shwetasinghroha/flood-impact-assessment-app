from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from utils import (
    IMAGE_SIZE,
    TARGET_CLASS_COLORS,
    preprocess_image,
    mask_to_rgb,
    overlay_mask_on_image,
    compute_flood_indicators,
    classify_severity_infrastructure_impact,
    generate_evidence_summary,
)


MODEL_PATH = Path("model/best_unet_advanced_section5b.pth")
NUM_CLASSES = 4
BASE_CHANNELS = 32
DROPOUT_P = 0.20


class DoubleConvGN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8, dropout_p: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, out_channels), num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, out_channels), num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetGN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 4, base_channels: int = 32, dropout_p: float = 0.2):
        super().__init__()

        self.enc1 = DoubleConvGN(in_channels, base_channels, dropout_p=0.0)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConvGN(base_channels, base_channels * 2, dropout_p=0.0)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConvGN(base_channels * 2, base_channels * 4, dropout_p=0.0)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConvGN(base_channels * 4, base_channels * 8, dropout_p=dropout_p / 2)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConvGN(base_channels * 8, base_channels * 16, dropout_p=dropout_p)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConvGN(base_channels * 16, base_channels * 8, dropout_p=dropout_p / 2)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConvGN(base_channels * 8, base_channels * 4, dropout_p=0.0)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConvGN(base_channels * 4, base_channels * 2, dropout_p=0.0)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConvGN(base_channels * 2, base_channels, dropout_p=0.0)

        self.out_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)


@dataclass
class FloodAssessmentOutput:
    predicted_mask: np.ndarray
    predicted_mask_rgb: np.ndarray
    overlay_image: np.ndarray
    indicators: Dict[str, float]
    severity: str
    evidence_summary: str
    resized_input_image: np.ndarray


class FloodAssessmentPredictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNetGN(
            in_channels=3,
            num_classes=NUM_CLASSES,
            base_channels=BASE_CHANNELS,
            dropout_p=DROPOUT_P,
        ).to(self.device)

        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def predict(self, pil_image: Image.Image, image_name: str = "uploaded_image") -> FloodAssessmentOutput:
        resized_image, model_input = preprocess_image(pil_image, image_size=IMAGE_SIZE)

        input_tensor = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.model(input_tensor)
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        pred_mask_rgb = mask_to_rgb(pred_mask, TARGET_CLASS_COLORS)
        overlay = overlay_mask_on_image(resized_image, pred_mask, TARGET_CLASS_COLORS, alpha=0.40)

        indicators = compute_flood_indicators(pred_mask)
        severity = classify_severity_infrastructure_impact(indicators)
        summary = generate_evidence_summary(image_name, indicators, severity)

        return FloodAssessmentOutput(
            predicted_mask=pred_mask,
            predicted_mask_rgb=pred_mask_rgb,
            overlay_image=overlay,
            indicators=indicators,
            severity=severity,
            evidence_summary=summary,
            resized_input_image=resized_image,
        )