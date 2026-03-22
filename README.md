# 🏦 Credit Decision Engine: Dual-Layer ML Framework

[](https://www.python.org/downloads/)
[](https://www.docker.com/)
[](https://streamlit.io/)

Sebuah sistem penilaian kredit (*credit scoring*) tingkat produksi yang menggabungkan presisi model non-linear dengan transparansi penjelasan model linear. Proyek ini dikembangkan untuk memprediksi risiko gagal bayar nasabah menggunakan dataset Home Credit dengan pendekatan **Challenger Model Architecture**.

-----

## 🏗️ System Architecture

Sistem ini menggunakan dua lapisan model untuk memastikan keputusan yang diambil tidak hanya akurat, tetapi juga dapat dijelaskan (*explainable*):

1.  **Layer 1: Decision Maker (LightGBM)**
      * Bertugas memberikan prediksi probabilitas gagal bayar.
      * Menggunakan algoritma *Gradient Boosting* untuk menangkap hubungan non-linear yang kompleks antara fitur keuangan nasabah.
2.  **Layer 2: Dynamic Explainer (Logistic Regression)**
      * Bertugas memberikan alasan di balik penolakan (*rejection reason*).
      * Menghitung dampak fitur secara dinamis untuk memberikan transparansi kepada nasabah dan regulator.

-----

## 📊 Model Performance

Berdasarkan pengujian pada *test set* yang tidak terlihat sebelumnya, model ini mencapai metrik stabilitas yang sangat tinggi:

| Metric | Value | Note |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.8525** | Menunjukkan kemampuan diskriminasi yang sangat baik. |
| **Recall (Bad Class)** | **82.19%** | Meminimalisir risiko kerugian bank dari nasabah berisiko. |
| **Stability Gap** | **0.0371** | Perbedaan minimal antara performa Train dan Test (Very Stable). |
| **Decision Threshold** | **0.69** | Titik optimal untuk menyeimbangkan *profitability* dan *risk*. |

-----

## 🛠️ Key Features (The Top 10)

Model ini dipangkas menjadi 10 fitur paling berpengaruh untuk menjaga stabilitas sistem dan memudahkan proses audit:

  * `mths_since_issue_d`: Durasi sejak pinjaman diberikan.
  * `mths_since_last_pymnt_d`: **(Rank 2)** Bulan sejak pembayaran terakhir.
  * `annual_inc`: **(Rank 3)** Pendapatan tahunan nasabah.
  * `dti`: Rasio utang terhadap pendapatan.
  * `revol_util`: Penggunaan limit kartu kredit.
  * *Dan 5 fitur teknis lainnya (Lihat dokumentasi API).*

-----

## 🧩 Robustness & Analysis (Edge Cases)

Sistem ini telah diuji terhadap skenario ekstrem untuk memastikan integritas logika:

### 1\. The "Cliff Effect" (4 vs 5 Months)

Melalui analisis **Partial Dependence Plot (PDP)**, ditemukan adanya lompatan risiko non-linear. Nasabah dengan keterlambatan 4 bulan memiliki risiko rendah (**\~10%**), namun melompat drastis ke **\~70%** pada bulan ke-5. Hal ini mencerminkan kebijakan *default* perbankan di dunia nyata.

### 2\. Outlier Mitigation

Sistem dilengkapi dengan fungsi *clipping* dan *log-transformation* untuk menangani data input yang tidak masuk akal (misal: Gaji \> 1 Miliar), memastikan bahwa pemicu penolakan tetap objektif dan tidak didominasi oleh satu fitur raksasa.

$$Impact_i = \log(1 + \text{Feature Value}_i) \times \text{Coefficient}_i$$

-----

## 🚀 Quick Start

### Running with Docker

Aplikasi ini sudah dikontainerisasi agar dapat berjalan di lingkungan mana pun tanpa konflik dependensi.

```bash
# Build image
docker build -t credit-scoring-app .

# Run container
docker run -p 8000:8000 credit-scoring-app
```

### Accessing the UI (Streamlit)

Untuk antarmuka yang lebih ramah pengguna:

```bash
pip install -r requirements.txt
streamlit run app.py
```

-----

## 👨‍💻 About the Author

**Muhammad Azka Ayubi**
Information Systems Graduate | FILKOM Universitas Brawijaya

  * **GPA**: 3.84/4.00
  * **Focus**: Data Science, Business Analysis, & UI/UX Design.

-----

> **Insight Wit**: Dokumen ini membuktikan bahwa model saya tidak bisa "disogok" gaji kuadriliun; kalau Anda telat bayar 5 bulan, sistem tetap bilang **Rejected**\!