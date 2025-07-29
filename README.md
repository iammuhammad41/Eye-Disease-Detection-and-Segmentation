# Retinal Disease Classification

## 📁 Data

- **Train images**:  
  `/kaggle/input/retinal-disease-classification/Training_Set/Training_Set/Training`
- **Validation images**:  
  `/kaggle/input/retinal-disease-classification/Evaluation_Set/Evaluation_Set/Validation`
- **Test images**:  
  `/kaggle/input/retinal-disease-classification/Test_Set/Test_Set/Test`

- **Labels CSVs**:  
  - `RFMiD_Training_Labels.csv`  
  - `RFMiD_Validation_Labels.csv`  
  - `RFMiD_Testing_Labels.csv`

Each label file contains an `ID` column matching `ID.png` and a `Disease_Risk` target (0 = low risk, 1 = high risk).

---

## ⚙️ Requirements

```bash
pip install \
  numpy pandas opencv-python matplotlib seaborn \
  plotly scikit-learn tensorflow
````

> All imports are included at the top of the notebook.

---

## 🧠 Model

* **Architecture**:

  * 3× Conv2D + MaxPooling
  * Flatten → Dense(512) + Dropout(0.5) → Dense(1, sigmoid)
* **Input size**: 224 × 224 × 3
* **Loss**: Binary cross‑entropy
* **Optimizer**: Adam
* **Metrics**: Accuracy

---

## 📈 Results

* **Test Accuracy**: \~86%
* **Precision / Recall / F1** printed in the classification report
* **Confusion Matrix** visualized with Plotly
* **Training curves** for loss & accuracy over 20 epochs


