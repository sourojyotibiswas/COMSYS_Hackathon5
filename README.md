---
# 🧠 Face Analysis Project: Gender Classification & Robust Face Recognition
---

## **1. Task A – Gender Classification**

### **1.1 🎯 Objective**

Build a high-accuracy deep learning model to classify face images as **Male** or **Female**, even under natural variations in lighting, pose, and image quality.

---

### **1.2 🛠️ Setup Instructions**

#### Dataset Structure:

```
Task_A/
├── train/
│   ├── male/
│   └── female/
└── val/
    ├── male/
    └── female/
```

#### Colab Environment Setup:

```python
from google.colab import drive
drive.mount('/content/drive')

!pip install torch torchvision matplotlib seaborn --quiet
```

---

### **1.3 🧪 Approach**

1. **Transfer Learning with ResNet18**

   - Load pretrained ResNet18 from `torchvision.models`.
   - Replace the final FC layer with `nn.Linear(512, 2)` for binary output.

2. **Robust Data Augmentation**

   - Techniques used:

     - `Resize`, `HorizontalFlip`, `ColorJitter`, `RandomRotation`
     - Optional: `GaussianBlur`

   - Applied only on the training set to improve generalization.

3. **Training Strategy**

   - Loss Function: `CrossEntropyLoss`
   - Optimizer: `Adam` with learning rate `1e-4`
   - Epochs: `10`, Batch size: `32`

4. **Evaluation Metrics**

   - Accuracy, Precision, Recall, F1-Score
   - Full classification report using `sklearn`
   - Confusion matrix visualization using `seaborn`

---

### **1.4 🧱 Model Architecture**

```
Input Image → [224x224 RGB]
    ↓
Pretrained ResNet18 Backbone
    ↓
Modified Fully Connected Layer (512 → 2)
    ↓
Softmax → Gender Prediction (Male / Female)
```

---

### **1.5 ✅ Key Innovations**

- Lightweight architecture for fast training and high performance.
- Strong augmentation pipeline to simulate real-world image variability.
- End-to-end model training, evaluation, visualization, and saving.

---

### **1.6 💾 Output**

- Trained model file: `model_resnet18_task_a.pth`
- Output includes:

  - Classification report
  - Confusion matrix
  - Gender-wise performance scores

---

## **2. Task B – Face Recognition under Challenging Visual Conditions**

### **2.1 🎯 Objective**

Train a deep learning model to recognize individuals from a fixed identity set, even when the faces are visually degraded (e.g., **blurred, foggy, sunny, rainy, low-light**).

---

### **2.2 🛠️ Setup Instructions**

#### Dataset Structure:

```
Task_B/
├── train/
│   └── <Person Name>/
│       ├── <person>.jpg
│       └── Distortion/
│           ├── <person>_blurred.jpg
│           ├── <person>_foggy.jpg
│           └── ...
├── val/
│   └── <Person Name>/
        ├── <person>.jpg
        └── Distortion/
            ├── <person>_lowlight.jpg
            └── ...
```

#### Colab Setup:

```python
from google.colab import drive
drive.mount('/content/drive')

dataset_path = "/content/drive/MyDrive/Task_B"
```

---

### **2.3 🧪 Approach**

1. **Custom Dataset Class**

   - Walks through all person folders and distortion subfolders.
   - Assigns class index to each person.

2. **Model Architecture**

   - Backbone: Pretrained `ResNet-50`
   - Final FC Layer: `2048 → num_classes`
   - Loss Function: `CrossEntropyLoss` with **class weights** to address imbalance.

3. **Training Strategy**

   - Optimizer: `Adam`, LR = `1e-4`
   - Scheduler: `StepLR(step_size=5, gamma=0.5)`
   - Batch size: `32`, Epochs: `10`
   - Checkpointing: Saves model with best macro F1 on validation.

4. **Visual Robustness**

   - Augmentation includes resized, brightened, noisy, and distorted samples.
   - Learns identity features invariant to environmental changes.

---

### **2.4 🧱 Model Pipeline**

```
Input Image (clean/distorted)
    ↓
Transform (resize, normalize, augment)
    ↓
Pretrained ResNet50 Backbone
    ↓
Modified FC Layer (2048 → num_classes)
    ↓
Softmax → Identity Prediction
```

---

### **2.5 ✅ Key Innovations**

- Integrated distortions during training for real-world robustness.
- Custom class balancing using `sklearn.compute_class_weight`.
- Modular training + evaluation scripts for portability.

---

### **2.6 💾 Output**

- Trained model file: `model_b.pth`
- Reports include:

  - Macro-averaged Accuracy, Precision, Recall, F1
  - Identity-wise classification report

---

### **2.7 📈 Evaluation Script**

Use `part_b_model_evaluation_only.py` to evaluate the model.

```bash
python part_b_model_evaluation_only.py \
  --data_path "/content/drive/MyDrive/Task_B" \
  --model_path "model_b.pth"
```

---
