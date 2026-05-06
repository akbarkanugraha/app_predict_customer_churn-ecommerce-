# 🛒 E-Commerce Customer Churn Prediction  
### End-to-End Machine Learning & Streamlit App

Aplikasi dan model machine learning untuk memprediksi **customer churn** pada platform e-commerce

Link apps : https://predict-customer-churn-27.streamlit.app/

---

## 📌 1. Overview Project

Proyek ini membangun sistem **end-to-end machine learning** untuk mengidentifikasi pelanggan yang berpotensi berhenti menggunakan layanan (churn).

Pipeline mencakup:

- Exploratory Data Analysis (EDA)  
- Feature Engineering  
- Model Benchmarking  
- Penalized Model (`class_weight`)  
- Hyperparameter Tuning  
- Evaluasi Test Set  
- SHAP Interpretability  
- Cost-Benefit Analysis  
- Model Deployment (Streamlit App)

Dataset berisi informasi pelanggan seperti:
- Demografi
- Perilaku transaksi
- Aktivitas aplikasi
- Kepuasan & komplain

---

## 🎯 2. Tujuan

1. Membangun model klasifikasi dengan **Recall tinggi**  
2. Mengidentifikasi faktor utama churn menggunakan **SHAP**  
3. Memberikan **insight actionable** untuk tim bisnis  

**Metric Utama:**  
➡️ Recall (menghindari False Negative / pelanggan churn yang tidak terdeteksi)

---

## 👥 3. Stakeholders

| Stakeholder | Peran |
|------------|------|
| Marketing / CRM | Menggunakan hasil prediksi untuk kampanye retensi |
| Customer Service | Follow-up pelanggan dengan risiko tinggi |

---

## 🧠 4. Model Result

**Model Terbaik:**  
- Support Vector Machine (SVM) (hasil tuning)

**Evaluasi (Test Set):**

| Metric | Value |
|--------|------|
| Recall (Test) | 1 |
| Precision (Test) | 0.15 |
| F1-Score (Test) | 0.264 |


---

## 💰 5. Cost-Benefit Analysis

Asumsi bisnis:

- Biaya retensi: **Rp 100.000 / pelanggan**
- Customer Lifetime Value: **Rp 1.500.000**
- Kerugian jika churn tidak terdeteksi: **Rp 1.500.000**

📊 Insight:
Model membantu mengurangi kerugian dengan mendeteksi pelanggan berisiko lebih awal.

---

## 🖥️ 6. Fitur Aplikasi (Streamlit)

| Fitur | Deskripsi |
|------|----------|
| Prediksi Tunggal | Input manual → hasil prediksi + probabilitas |
| Prediksi Batch | Upload CSV → hasil massal |
| Risk Level | Tinggi / Sedang / Rendah |
| Top Churners | 10 pelanggan paling berisiko |
| Download CSV | Export hasil prediksi |
| Template CSV | Format input siap pakai |

---

## 📋 7. Format Input CSV

Kolom yang diperlukan:

Tenure, PreferredLoginDevice, CityTier, WarehouseToHome,
PreferredPaymentMode, Gender, HourSpendOnApp, NumberOfDeviceRegistered,
PreferedOrderCat, SatisfactionScore, MaritalStatus, NumberOfAddress,
Complain, OrderAmountHikeFromlastYear, CouponUsed, OrderCount,
DaySinceLastOrder, CashbackAmount

---

## 📁 8. Struktur Project

project/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── data_ecommerce_customer_churn.csv
├── notebook/
│   └── capstone_churn_final.ipynb
└── models/
    └── best_churn_model.pkl

---

## ⚙️ 9. Cara Menjalankan

### ▶️ Jalankan Lokal

pip install -r requirements.txt

# pastikan model tersedia
mkdir -p models
cp /path/to/best_churn_model.pkl models/

streamlit run app.py
---

## 🧪 10. Generate Model dari Notebook

Di notebook:

import joblib  
joblib.dump(sv_best, 'models/best_churn_model.pkl')

Pastikan file ini tersedia sebelum menjalankan aplikasi.

---

## 💡 11. Rekomendasi

### 📊 Data
- Tambahkan fitur RFM (Recency, Frequency, Monetary)  
- Gunakan data interaksi (klik, wishlist, dll)

### 🤖 Model
- Retrain setiap 3 bulan  
- Monitor model drift  

### 📈 Bisnis
- Jalankan model secara batch mingguan  
- Integrasikan ke CRM system  
- Kirim campaign retensi 1–2 minggu sebelum churn  

---

## 🚀 12. Future Improvement

- Real-time prediction (API)
- Dashboard monitoring model
- Auto retraining pipeline
- Integration dengan marketing tools
