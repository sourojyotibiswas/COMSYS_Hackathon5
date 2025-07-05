## Part B: Face Recognition under Challenging Visual Conditions

### Training

Run the script to:

- Load the dataset from `/train` and `/val` directories
- Train a ResNet-50 classifier to recognize person identities
- Automatically include all visual distortions like blur, fog, rain, low light, etc.
- Handle class imbalance using weighted CrossEntropyLoss
- Save the best-performing model (`model_b.pth`) to your Google Drive or local path

> Make sure to update the `dataset_path` variable in the script to point to your dataset:
>
> ```python
> dataset_path = "/content/drive/MyDrive/Task_B"
> ```

---

### Evaluation Only

To load the saved model and evaluate it on both training and validation sets:

- Run `<file_name.py>`
- Set the correct path to `model_b.pth` and dataset folder via command-line or modify the script

**Outputs include:**

- Top-1 Accuracy
- Precision (macro)
- Recall (macro)
- F1 Score (macro)
- Full classification report for each identity
- Class distribution for both train and validation sets

Example command:

```bash
python <file_name.py> \
  --data_path "/content/drive/MyDrive/Task_B" \
  --model_path "model_b.pth"
```
