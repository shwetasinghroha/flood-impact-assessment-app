# Flood Impact Assessment App

This app performs preliminary flood impact assessment from post-flood UAV imagery.

## Features

- Upload a UAV image
- Predict flood-relevant mask
- Generate overlay
- Compute flood indicators:
  - Water %
  - Flooded Building %
  - Flooded Road %
  - Combined Flooded Infrastructure %
- Assign preliminary infrastructure-impact severity
- Generate evidence summary

## Repo structure

- `app.py` — Streamlit frontend
- `inference.py` — model loading and inference
- `utils.py` — preprocessing, indicators, severity, overlays
- `model/best_unet_advanced_section5b.pth` — trained checkpoint

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py