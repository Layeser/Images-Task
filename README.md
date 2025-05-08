# UTKFace Age & Gender Prediction

This project demonstrates how to train a deep learning model using the [UTKFace dataset](https://susanqq.github.io/UTKFace/) to **predict age (regression)** and **gender (classification)** from facial images. The model is built with PyTorch and uses a pre-trained EfficientNet-B0 as its backbone.

---

## Dataset

The **UTKFace** dataset contains over 20,000 face images with labels for:
- Age (0–116)
- Gender (0: Male, 1: Female)

Each image filename follows the format: `age_gender_race_date.jpg`.
---

## Model Architecture

- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
- **Age Regressor**: Fully connected layers with dropout + sigmoid
- **Gender Classifier**: Fully connected layers with dropout

```python
Input Image → EfficientNetB0 → Shared Features →
    ├── Age Regression Head (SmoothL1Loss)
    └── Gender Classification Head (BCEWithLogitsLoss)
