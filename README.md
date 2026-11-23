# Proyek Akhir Machine Learning: Prediksi Biaya Asuransi Kesehatan

Proyek akhir mata kuliah Machine Learning (Semester 5) di Universitas Trunojoyo Madura untuk memprediksi premi/biaya asuransi kesehatan perorangan menggunakan algoritma **Classification and Regression Trees (CART)** dengan hyperparameter tuning 5-Fold Cross Validation.

---

## 📁 Struktur Direktori Proyek

```text
project/
├── datasets/
│   └── insurance.csv                        # Dataset riwayat biaya medis (1.337 data unik)
│
├── model/                                   # 📁 KUMPULAN KODE & HASIL MACHINE LEARNING
│   ├── model_insurance_cart.ipynb           # Notebook utama penelitian (6 tahapan metodologi)
│   ├── train_model.py                       # Script Python pelatihan & evaluasi model CART
│   ├── app.py                               # Aplikasi web simulasi & alur interaktif (Flask)
│   ├── templates/
│   │   └── index.html                       # Antarmuka web (Visualisasi pohon interaktif SVG)
│   └── outputs/                             # Seluruh artefak model & visualisasi
│       ├── models/
│       │   └── best_cart_model.joblib       # Model CART optimal hasil GridSearchCV
│       └── visualizations/
│           ├── decision_tree_diagram.png    # Diagram arsitektur pohon keputusan CART
│           ├── feature_importance.png       # Bobot kepentingan fitur (smoker, bmi, age)
│           ├── actual_vs_predicted.png      # Scatter plot nilai aktual vs prediksi
│           ├── eda_distribution_charges.png # Distribusi target biaya (positively skewed)
│           └── eda_smoker_bmi_interaction.png # Interaksi non-linear perokok dan BMI
│
├── PROPOSAL_PROJECT_AKHIR_MACHINE_LEARNING_v2.docx # Dokumen proposal lengkap (Bab 1 - Bab 4)
├── PRESENTASI_PROJECT_AKHIR_MACHINE_LEARNING.pptx   # Slide presentasi seminar proposal
├── PROJECT AKHIR MACHINE LEARNING.pdf              # Panduan instruksi tugas dari dosen
└── README.md                                       # Panduan dokumentasi proyek
```

---

## 🚀 Cara Menjalankan Proyek

### 1. Eksplorasi Notebook Interaktif
Buka file `model/model_insurance_cart.ipynb` menggunakan Jupyter Lab, Jupyter Notebook, atau VS Code:
```bash
jupyter notebook model/model_insurance_cart.ipynb
```

### 2. Melatih Model & Memperbarui Visualisasi
Jalankan script pelatihan dari direktori proyek:
```bash
python model/train_model.py
```

### 3. Menjalankan Simulasi Web Interaktif
Aplikasi web menyediakan simulasi kalkulasi premi dan diagram alur pohon interaktif real-time:
```bash
python model/app.py
```
Buka browser pada alamat: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 📊 Hasil Evaluasi Model CART

| Model | Train $R^2$ | Test $R^2$ | Test MAE (USD) | Test RMSE (USD) |
| :--- | :---: | :---: | :---: | :---: |
| **CART Default** *(Unconstrained)* | 100.00% | 81.89% | $2,730.63 | $5,769.01 |
| **CART Tuned** *(Best Model: depth=4)* | **85.52%** | **89.72%** | **$2,621.31** | **$4,345.88** |

* Hyperparameter terbaik: `max_depth=4`, `min_samples_leaf=5`, `min_samples_split=2`.
* Tiga fitur paling berpengaruh: **Status Merokok (67.4%)**, **BMI (17.5%)**, dan **Usia (14.2%)**.
