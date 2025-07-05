# Face Analysis Under Real-World Visual Distortions

This project contains solutions for two face analysis tasks under challenging visual conditions:

---

## 🧩 Part A: Gender Classification

> Predict the gender (Male/Female) from facial images.

### Files:

- `part_a_model_training.ipynb`: Train a ResNet-18 classifier on the gender classification task
- `part_a_model_evaluation_only.ipynb`: Load the trained model and evaluate on validation data
- `README.md`: Setup instructions and usage for Part A

📁 Folder: [`part_a/`](./part_a/)

---

## 🧩 Part B: Face Recognition

> Recognize a person’s identity from face images affected by blur, fog, rain, low-light, glare, etc.

### Files:

- `part_b_model_training.ipynb`: Train a ResNet-50 classifier to recognize identities under visual distortions
- `part_b_model_evaluation_only.ipynb`: Evaluate the trained model on both training and validation datasets
- `README.md`: Instructions specific to Part B, including class imbalance handling

📁 Folder: [`part_b/`](./part_b/)

---

## 🛠️ Environment & Requirements

- Python 3.8+
- PyTorch
- torchvision
- scikit-learn
- PIL / Pillow
- tqdm
- Google Colab or GPU environment recommended

---

## 📁 Dataset Format (Expected)

Both Part A and Part B assume a dataset stored in Google Drive, mounted via:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```

Make sure to update `dataset_path` in the code as per your Drive location.

---

## 📦 Output

Each model saves the best checkpoint as:

- `<model_name>.pth`

You can re-use this file in the evaluation notebooks.

---

## 📑 Evaluation Metrics

Each part uses the following metrics (macro-averaged when applicable):

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Classification Report**

---

## ✅ Status

| Task                       | Model      | Status      |
| -------------------------- | ---------- | ----------- |
| Gender Classification      | ResNet-18  | ✅ Complete |
| Face Recognition           | ResNet-50  | ✅ Complete |
| Visual Distortion Handling | Both Tasks | ✅ Included |
| Class Imbalance Handling   | Part B     | ✅ Included |

---

> Built with robustness in mind, ensuring face models work reliably in real-world degraded environments.
```
