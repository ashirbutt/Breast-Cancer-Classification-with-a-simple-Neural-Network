
# Breast Cancer Classification with Neural Networks 🧠🔬

This project implements a deep learning-based approach to classify breast cancer as **Malignant** or **Benign** using the **Wisconsin Breast Cancer Dataset**. The classification model is built using a **Neural Network (NN)** implemented with **Keras/TensorFlow**.

## 📌 Objective
To build and train a neural network that can accurately detect whether a tumor is malignant or benign based on various diagnostic features of cell nuclei.

---

## 📂 Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Attributes**: 30 numeric features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
- **Target**: 
  - **0**: Malignant  
  - **1**: Benign

---

## 🚀 Technologies Used

- Python 🐍
- Pandas
- NumPy
- Matplotlib & Seaborn (for EDA)
- Scikit-learn
- TensorFlow & Keras (Neural Network Model)

---

## 📊 Exploratory Data Analysis (EDA)

- Null values check ✅  
- Class balance visualization  
- Correlation heatmap  
- Feature distribution plots  

---

## 🧠 Model Architecture

- **Input Layer**: 30 neurons (features)
- **Hidden Layers**: 2 hidden layers (ReLU activation)
- **Output Layer**: 1 neuron with **sigmoid** activation (binary classification)

```python
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=30))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

- **Loss**: Binary Crossentropy  
- **Optimizer**: Adam  
- **Metrics**: Accuracy  

---

## 📈 Training & Evaluation

- **Train-Test Split**: 80/20  
- **Epochs**: 100  
- **Batch Size**: 16  
- **Validation**: Done during training  
- **Evaluation Metrics**:
  - Accuracy
  - Confusion Matrix
  - ROC Curve & AUC
  - Classification Report

---

## ✅ Results

- **Training Accuracy**: ~99%  
- **Test Accuracy**: ~98%  
- Model generalizes well with minimal overfitting.

---

## 📌 How to Run

1. Clone the repo
2. Install required libraries:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook:
   ```bash
   jupyter notebook Copy_of_DL_Project_1_Breast_Cancer_Classification_with_NN.ipynb
   ```

---

## 📁 File Structure

```
├── Copy_of_DL_Project_1_Breast_Cancer_Classification_with_NN.ipynb
├── README.md
└── requirements.txt
```

---

## 📚 References

- UCI Breast Cancer Dataset
- TensorFlow Documentation
- Keras Tutorials

---

## 🙌 Acknowledgements

This project is part of a Deep Learning coursework exercise focused on practical model building and evaluation.
