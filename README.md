# Semantic Segmentation using DINOv2 and U-Net

This project performs semantic segmentation on Pascal VOC 2012 and AeroScapes datasets using a hybrid architecture that combines a pretrained DINOv2 Vision Transformer (ViT) encoder with a U-Net-style decoder. The model is optimized using Lovasz Softmax Loss to improve mean Intersection-over-Union (mIoU), with selective fine-tuning applied to increase training efficiency.

## Dataset Used

The experiments are conducted on two distinct datasets with ground-view and aerial-view segmentation challenges:

### 1. Pascal VOC 2012
- **Classes:** 21 (20 object classes + background)
- **Image Resolution:** Resized to 518×518
- **Dataset Link:** [Pascal VOC on Kaggle](https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012)

### 2. AeroScapes (UAV Aerial Dataset)
- **Classes:** 11
- **Image Resolution:** Resized to 518×518
- **Dataset Link:** [AeroScapes on Kaggle](https://www.kaggle.com/datasets/kooaslansefat/uav-segmentation-aeroscapes/data)

---

## Model Architectures

### Pascal VOC Model

| Component | Description |
|-----------|-------------|
| **Backbone** | `vit_base_patch14_dinov2.lvd142m` via `timm` |
| **Decoder** | U-Net style decoder with ConvBlocks |
| **Tuning Strategy** | Only last transformer block (`blocks.11`) and decoder are trained |
| **Loss Function** | Lovasz Softmax Loss (suitable for optimizing IoU) |

### AeroScapes Model

| Component | Description |
|-----------|-------------|
| **Backbone** | `vit_base_patch14_dinov2` via `timm` |
| **Decoder** | Transposed convolution upsampling with center cropping |
| **Tuning Strategy** | Backbone fully frozen; only decoder is trained |
| **Loss Function** | CrossEntropyLoss |

---

## Loss Functions

- **Pascal VOC:** Lovasz Softmax Loss
- **AeroScapes:** Cross Entropy Loss

---

## Evaluation Metrics

Evaluation is based on:
- **Pixel-Wise Accuracy:** Ratio of correctly classified pixels to total pixels.
- **Mean IoU (mIoU):** Average Intersection-over-Union across all classes.

### Pascal VOC

| Epoch | Train Loss | Val Accuracy | Val mIoU |
|-------|------------|--------------|----------|
| 1     | 0.6927     | 0.7481       | 0.0868   |
| 2     | 0.5166     | 0.8150       | 0.1073   |
| 3     | 0.4497     | 0.8839       | 0.1196   |
| 4     | 0.3999     | 0.8839       | 0.1202   |
| 5     | 0.3499     | 0.9125       | 0.1290   |
| 6     | 0.3193     | 0.9350       | 0.1361   |
| 7     | 0.2882     | 0.9426       | 0.1376   |
| 8     | 0.2659     | 0.9368       | 0.1373   |

### AeroScapes (Aerial Dataset)

| Epoch | Training Loss | Validation mIoU | Pixel-Wise Accuracy |
|-------|----------------|------------------|----------------------|
| 1     | 1.1374         | 0.199            | 0.6408               |
| 2     | 0.4045         | 0.322            | 0.8590               |
| 4     | 0.2148         | 0.354            | 0.8802               |
| 6     | 0.1750         | 0.3627           | 0.8837               |
| 7     | 0.1617         | 0.3594           | 0.8853               |
| 8     | 0.1523         | 0.3683           | 0.8846               |
| 9     | 0.1464         | 0.3618           | 0.8870               |
| 10    | 0.1372         | 0.3745           | 0.8824               |
| 11    | 0.1329         | 0.3823           | 0.8890               |

---

## Training Settings

| Setting         | Pascal VOC         | AeroScapes      |
|-----------------|--------------------|------------------|
| **Optimizer**   | AdamW              | Adam             |
| **Learning Rate**| 1e-4              | 1e-4             |
| **Batch Size**  | 2                  | 2                |
| **Epochs**      | 8                  | 11               |

---

## Resources

- **Notebooks**:
  - [Pascal VOC Training Notebook](./Notebook_Pascal_VOC_2012.ipynb)
  - [AeroScapes Training Notebook](./Notebook_Arial_Dataset.ipynb)

- **Project Presentation**:
  - [Watch this video on YouTube](https://youtu.be/h-K6XyY-x-w?si=bvMmiE8Rc742R1_K)

- **Project Reports**:
  - [Pascal VOC Project Report](./Semantic Segmentation Using DINOv2 ViT and U-Net on Pascal VOC 2012.pdf)
  - [AeroScapes Project Report](./Semantic Segmentation Using DINO ViT and U-Net on Aerial Dataset.pdf)
---


